// SPDX-License-Identifier: GPL-2.0-or-later

#include <linux/sched/task.h>
#include <linux/sched/signal.h>
#include <linux/freezer.h>

#include "futex.h"

/*
 * READ this before attempting to hack on futexes!
 *
 * Basic futex operation and ordering guarantees
 * =============================================
 *
 * The waiter reads the futex value in user space and calls
 * futex_wait(). This function computes the hash bucket and acquires
 * the hash bucket lock. After that it reads the futex user space value
 * again and verifies that the data has not changed. If it has not changed
 * it enqueues itself into the hash bucket, releases the hash bucket lock
 * and schedules.
 *
 * The waker side modifies the user space value of the futex and calls
 * futex_wake(). This function computes the hash bucket and acquires the
 * hash bucket lock. Then it looks for waiters on that futex in the hash
 * bucket and wakes them.
 *
 * In futex wake up scenarios where no tasks are blocked on a futex, taking
 * the hb spinlock can be avoided and simply return. In order for this
 * optimization to work, ordering guarantees must exist so that the waiter
 * being added to the list is acknowledged when the list is concurrently being
 * checked by the waker, avoiding scenarios like the following:
 *
 * CPU 0                               CPU 1
 * val = *futex;
 * sys_futex(WAIT, futex, val);
 *   futex_wait(futex, val);
 *   uval = *futex;
 *                                     *futex = newval;
 *                                     sys_futex(WAKE, futex);
 *                                       futex_wake(futex);
 *                                       if (queue_empty())
 *                                         return;
 *   if (uval == val)
 *      lock(hash_bucket(futex));
 *      queue();
 *     unlock(hash_bucket(futex));
 *     schedule();
 *
 * This would cause the waiter on CPU 0 to wait forever because it
 * missed the transition of the user space value from val to newval
 * and the waker did not find the waiter in the hash bucket queue.
 *
 * The correct serialization ensures that a waiter either observes
 * the changed user space value before blocking or is woken by a
 * concurrent waker:
 *
 * CPU 0                                 CPU 1
 * val = *futex;
 * sys_futex(WAIT, futex, val);
 *   futex_wait(futex, val);
 *
 *   waiters++; (a)
 *   smp_mb(); (A) <-- paired with -.
 *                                  |
 *   lock(hash_bucket(futex));      |
 *                                  |
 *   uval = *futex;                 |
 *                                  |        *futex = newval;
 *                                  |        sys_futex(WAKE, futex);
 *                                  |          futex_wake(futex);
 *                                  |
 *                                  `--------> smp_mb(); (B)
 *   if (uval == val)
 *     queue();
 *     unlock(hash_bucket(futex));
 *     schedule();                         if (waiters)
 *                                           lock(hash_bucket(futex));
 *   else                                    wake_waiters(futex);
 *     waiters--; (b)                        unlock(hash_bucket(futex));
 *
 * Where (A) orders the waiters increment and the futex value read through
 * atomic operations (see futex_hb_waiters_inc) and where (B) orders the write
 * to futex and the waiters read (see futex_hb_waiters_pending()).
 *
 * This yields the following case (where X:=waiters, Y:=futex):
 *
 *	X = Y = 0
 *
 *	w[X]=1		w[Y]=1
 *	MB		MB
 *	r[Y]=y		r[X]=x
 *
 * Which guarantees that x==0 && y==0 is impossible; which translates back into
 * the guarantee that we cannot both miss the futex variable change and the
 * enqueue.
 *
 * Note that a new waiter is accounted for in (a) even when it is possible that
 * the wait call can return error, in which case we backtrack from it in (b).
 * Refer to the comment in futex_q_lock().
 *
 * Similarly, in order to account for waiters being requeued on another
 * address we always increment the waiters for the destination bucket before
 * acquiring the lock. It then decrements them again  after releasing it -
 * the code that actually moves the futex(es) between hash buckets (requeue_futex)
 * will do the additional required waiter count housekeeping. This is done for
 * double_lock_hb() and double_unlock_hb(), respectively.
 */

/*
 * The hash bucket lock must be held when this is called.
 * Afterwards, the futex_q must not be accessed. Callers
 * must ensure to later call wake_up_q() for the actual
 * wakeups to occur.
 */
void futex_wake_mark(struct wake_q_head *wake_q, struct futex_q *q)
{
	struct task_struct *p = q->task;

	if (WARN(q->pi_state || q->rt_waiter, "refusing to wake PI futex\n"))
		return;

	get_task_struct(p);
	__futex_unqueue(q);
	/*
	 * The waiting task can free the futex_q as soon as q->lock_ptr = NULL
	 * is written, without taking any locks. This is possible in the event
	 * of a spurious wakeup, for example. A memory barrier is required here
	 * to prevent the following store to lock_ptr from getting ahead of the
	 * plist_del in __futex_unqueue().
	 */
	smp_store_release(&q->lock_ptr, NULL);

	/*
	 * Queue the task for later wakeup for after we've released
	 * the hb->lock.
	 */
	wake_q_add_safe(wake_q, p);
}

/*
 * Wake up waiters matching bitset queued on this futex (uaddr).
 */
/*
 * 唤醒在指定 futex (uaddr) 上排队的与 bitset 匹配的等待者 (waiters)。
 * 
 * 参数:
 * uaddr  - 指向用户空间的 futex 地址
 * flags  - 指定 futex 的一些行为，比如是否共享
 * nr_wake - 要唤醒的等待者数量
 * bitset - 用于匹配等待者的位集
 * 
 * 返回值:
 * 成功时返回实际唤醒的等待者数量，失败时返回负数错误码。
 */
int futex_wake(u32 __user *uaddr, unsigned int flags, int nr_wake, u32 bitset)
{
	struct futex_hash_bucket *hb;//futex锁对应的hash桶
	struct futex_q *this, *next;// 遍历 futex 等待队列的指针
	union futex_key key = FUTEX_KEY_INIT;// futex 键值，用于标识 futex
	int ret;// 返回值，记录唤醒的任务数量或错误码
	DEFINE_WAKE_Q(wake_q);// 定义用于存储将要唤醒的任务队列

	if (!bitset)
		return -EINVAL;
	// 获取 futex 键值，根据 futex 的共享标志 (flags) 来获取 key 值
	ret = get_futex_key(uaddr, flags & FLAGS_SHARED, &key, FUTEX_READ);
	if (unlikely(ret != 0))
		return ret;
	// 通过 key 来查找对应的哈希桶 (hash bucket)
	hb = futex_hash(&key);

	/* 确保哈哈希桶中有对应的等待者，没有的话就没必要再唤醒 */
	if (!futex_hb_waiters_pending(hb))
		return ret;
	/*通过对哈希桶自旋锁上锁，来保证没有其他线程对该哈西桶进行修改*/
	spin_lock(&hb->lock);

	/*遍历哈希桶链表中的等待队列*/
	plist_for_each_entry_safe(this, next, &hb->chain, list) {
		/* 判断当前进程的key和给定的key是否一致*/
		if (futex_match (&this->key, &key)) {
			/*带有优先级或实时标记的等待者不唤醒*/
			if (this->pi_state || this->rt_waiter) {
				ret = -EINVAL;
				break;
			}

			/* Check if one of the bits is set in both bitsets */
			if (!(this->bitset & bitset))
				continue;
			/*将符合条件的等待者标记为可唤醒状态*/
			futex_wake_mark(&wake_q, this);
			if (++ret >= nr_wake)
				break;
		}
	}
	/*释放哈希桶锁*/
	spin_unlock(&hb->lock);
	/*唤醒已经标记的任务*/
	wake_up_q(&wake_q);
	return ret;
}

static int futex_atomic_op_inuser(unsigned int encoded_op, u32 __user *uaddr)
{
	unsigned int op =	  (encoded_op & 0x70000000) >> 28;
	unsigned int cmp =	  (encoded_op & 0x0f000000) >> 24;
	int oparg = sign_extend32((encoded_op & 0x00fff000) >> 12, 11);
	int cmparg = sign_extend32(encoded_op & 0x00000fff, 11);
	int oldval, ret;

	if (encoded_op & (FUTEX_OP_OPARG_SHIFT << 28)) {
		if (oparg < 0 || oparg > 31) {
			char comm[sizeof(current->comm)];
			/*
			 * kill this print and return -EINVAL when userspace
			 * is sane again
			 */
			pr_info_ratelimited("futex_wake_op: %s tries to shift op by %d; fix this program\n",
					get_task_comm(comm, current), oparg);
			oparg &= 31;
		}
		oparg = 1 << oparg;
	}

	pagefault_disable();
	ret = arch_futex_atomic_op_inuser(op, oparg, &oldval, uaddr);
	pagefault_enable();
	if (ret)
		return ret;

	switch (cmp) {
	case FUTEX_OP_CMP_EQ:
		return oldval == cmparg;
	case FUTEX_OP_CMP_NE:
		return oldval != cmparg;
	case FUTEX_OP_CMP_LT:
		return oldval < cmparg;
	case FUTEX_OP_CMP_GE:
		return oldval >= cmparg;
	case FUTEX_OP_CMP_LE:
		return oldval <= cmparg;
	case FUTEX_OP_CMP_GT:
		return oldval > cmparg;
	default:
		return -ENOSYS;
	}
}

/*
 * Wake up all waiters hashed on the physical page that is mapped
 * to this virtual address:
 */
int futex_wake_op(u32 __user *uaddr1, unsigned int flags, u32 __user *uaddr2,
		  int nr_wake, int nr_wake2, int op)
{
	union futex_key key1 = FUTEX_KEY_INIT, key2 = FUTEX_KEY_INIT;
	struct futex_hash_bucket *hb1, *hb2;
	struct futex_q *this, *next;
	int ret, op_ret;
	DEFINE_WAKE_Q(wake_q);

retry:
	ret = get_futex_key(uaddr1, flags & FLAGS_SHARED, &key1, FUTEX_READ);
	if (unlikely(ret != 0))
		return ret;
	ret = get_futex_key(uaddr2, flags & FLAGS_SHARED, &key2, FUTEX_WRITE);
	if (unlikely(ret != 0))
		return ret;

	hb1 = futex_hash(&key1);
	hb2 = futex_hash(&key2);

retry_private:
	double_lock_hb(hb1, hb2);
	op_ret = futex_atomic_op_inuser(op, uaddr2);
	if (unlikely(op_ret < 0)) {
		double_unlock_hb(hb1, hb2);

		if (!IS_ENABLED(CONFIG_MMU) ||
		    unlikely(op_ret != -EFAULT && op_ret != -EAGAIN)) {
			/*
			 * we don't get EFAULT from MMU faults if we don't have
			 * an MMU, but we might get them from range checking
			 */
			ret = op_ret;
			return ret;
		}

		if (op_ret == -EFAULT) {
			ret = fault_in_user_writeable(uaddr2);
			if (ret)
				return ret;
		}

		cond_resched();
		if (!(flags & FLAGS_SHARED))
			goto retry_private;
		goto retry;
	}

	plist_for_each_entry_safe(this, next, &hb1->chain, list) {
		if (futex_match (&this->key, &key1)) {
			if (this->pi_state || this->rt_waiter) {
				ret = -EINVAL;
				goto out_unlock;
			}
			futex_wake_mark(&wake_q, this);
			if (++ret >= nr_wake)
				break;
		}
	}

	if (op_ret > 0) {
		op_ret = 0;
		plist_for_each_entry_safe(this, next, &hb2->chain, list) {
			if (futex_match (&this->key, &key2)) {
				if (this->pi_state || this->rt_waiter) {
					ret = -EINVAL;
					goto out_unlock;
				}
				futex_wake_mark(&wake_q, this);
				if (++op_ret >= nr_wake2)
					break;
			}
		}
		ret += op_ret;
	}

out_unlock:
	double_unlock_hb(hb1, hb2);
	wake_up_q(&wake_q);
	return ret;
}

static long futex_wait_restart(struct restart_block *restart);

/**
 * futex_wait_queue() - futex_queue() and wait for wakeup, timeout, or signal
 * @hb:		the futex hash bucket, must be locked by the caller
 * @q:		the futex_q to queue up on
 * @timeout:	the prepared hrtimer_sleeper, or null for no timeout
 */

/*1.将当前线程挂到指定的哈希桶中
 *2.将当前线程状态设置为阻塞态
 *3.设置一个定时器并使用schedule让当前cpu去调度*/
void futex_wait_queue(struct futex_hash_bucket *hb, struct futex_q *q,
			    struct hrtimer_sleeper *timeout)
{
    /*
     * 在另一个线程唤醒当前线程之前，确保任务状态已经被设置为可中断的阻塞态。
     * set_current_state() 使用了 smp_store_mb() 来保证内存屏障，
     * futex_queue() 会调用 spin_unlock() 完成对 Futex 队列的同步。
     */
	/*将当前的线程设置为阻塞态*/
	set_current_state(TASK_INTERRUPTIBLE|TASK_FREEZABLE);
	/*将futex等待节点放入对应的hash桶中;通过__futex_queue实现的*/
	futex_queue(q, hb);

	/* 如果指定了超时参数,启动定时器 */
	if (timeout)
		hrtimer_sleeper_start_expires(timeout, HRTIMER_MODE_ABS);

    /*
     * 如果 Futex 等待队列条目还没有被移出哈希链表，则说明线程还没有被唤醒。
     * 此时继续等待并调用调度函数 schedule() 进入睡眠状态。
     */
	if (likely(!plist_node_empty(&q->list))) {
        /*
         * 如果定时器已经过期，当前线程会被标记为需要重新调度。
         * 如果没有超时限制或者定时器尚未过期，调用 `schedule()` 进入睡眠。
         */
		if (!timeout || timeout->task)
			schedule();//进入睡眠状态
	}
	/****************************************************************/
	/*                      线程被阻塞,等待被唤醒                     */
	/****************************************************************/
	/*唤醒后，将当前线程的状态恢复为运行状态*/
	__set_current_state(TASK_RUNNING);
}

/**
 * unqueue_multiple - Remove various futexes from their hash bucket
 * @v:	   The list of futexes to unqueue
 * @count: Number of futexes in the list
 *
 * Helper to unqueue a list of futexes. This can't fail.
 *
 * Return:
 *  - >=0 - Index of the last futex that was awoken;
 *  - -1  - No futex was awoken
 */
static int unqueue_multiple(struct futex_vector *v, int count)
{
	int ret = -1, i;

	for (i = 0; i < count; i++) {
		if (!futex_unqueue(&v[i].q))
			ret = i;
	}

	return ret;
}

/**
 * futex_wait_multiple_setup - Prepare to wait and enqueue multiple futexes
 * @vs:		The futex list to wait on
 * @count:	The size of the list
 * @woken:	Index of the last woken futex, if any. Used to notify the
 *		caller that it can return this index to userspace (return parameter)
 *
 * Prepare multiple futexes in a single step and enqueue them. This may fail if
 * the futex list is invalid or if any futex was already awoken. On success the
 * task is ready to interruptible sleep.
 *
 * Return:
 *  -  1 - One of the futexes was woken by another thread
 *  -  0 - Success
 *  - <0 - -EFAULT, -EWOULDBLOCK or -EINVAL
 */
static int futex_wait_multiple_setup(struct futex_vector *vs, int count, int *woken)
{
	struct futex_hash_bucket *hb;
	bool retry = false;
	int ret, i;
	u32 uval;

	/*
	 * Enqueuing multiple futexes is tricky, because we need to enqueue
	 * each futex on the list before dealing with the next one to avoid
	 * deadlocking on the hash bucket. But, before enqueuing, we need to
	 * make sure that current->state is TASK_INTERRUPTIBLE, so we don't
	 * lose any wake events, which cannot be done before the get_futex_key
	 * of the next key, because it calls get_user_pages, which can sleep.
	 * Thus, we fetch the list of futexes keys in two steps, by first
	 * pinning all the memory keys in the futex key, and only then we read
	 * each key and queue the corresponding futex.
	 *
	 * Private futexes doesn't need to recalculate hash in retry, so skip
	 * get_futex_key() when retrying.
	 */
retry:
	for (i = 0; i < count; i++) {
		if ((vs[i].w.flags & FUTEX_PRIVATE_FLAG) && retry)
			continue;

		ret = get_futex_key(u64_to_user_ptr(vs[i].w.uaddr),
				    !(vs[i].w.flags & FUTEX_PRIVATE_FLAG),
				    &vs[i].q.key, FUTEX_READ);

		if (unlikely(ret))
			return ret;
	}

	set_current_state(TASK_INTERRUPTIBLE|TASK_FREEZABLE);

	for (i = 0; i < count; i++) {
		u32 __user *uaddr = (u32 __user *)(unsigned long)vs[i].w.uaddr;
		struct futex_q *q = &vs[i].q;
		u32 val = (u32)vs[i].w.val;

		hb = futex_q_lock(q);
		ret = futex_get_value_locked(&uval, uaddr);

		if (!ret && uval == val) {
			/*
			 * The bucket lock can't be held while dealing with the
			 * next futex. Queue each futex at this moment so hb can
			 * be unlocked.
			 */
			futex_queue(q, hb);
			continue;
		}

		futex_q_unlock(hb);
		__set_current_state(TASK_RUNNING);

		/*
		 * Even if something went wrong, if we find out that a futex
		 * was woken, we don't return error and return this index to
		 * userspace
		 */
		*woken = unqueue_multiple(vs, i);
		if (*woken >= 0)
			return 1;

		if (ret) {
			/*
			 * If we need to handle a page fault, we need to do so
			 * without any lock and any enqueued futex (otherwise
			 * we could lose some wakeup). So we do it here, after
			 * undoing all the work done so far. In success, we
			 * retry all the work.
			 */
			if (get_user(uval, uaddr))
				return -EFAULT;

			retry = true;
			goto retry;
		}

		if (uval != val)
			return -EWOULDBLOCK;
	}

	return 0;
}

/**
 * futex_sleep_multiple - Check sleeping conditions and sleep
 * @vs:    List of futexes to wait for
 * @count: Length of vs
 * @to:    Timeout
 *
 * Sleep if and only if the timeout hasn't expired and no futex on the list has
 * been woken up.
 */
static void futex_sleep_multiple(struct futex_vector *vs, unsigned int count,
				 struct hrtimer_sleeper *to)
{
	if (to && !to->task)
		return;

	for (; count; count--, vs++) {
		if (!READ_ONCE(vs->q.lock_ptr))
			return;
	}

	schedule();
}

/**
 * futex_wait_multiple - Prepare to wait on and enqueue several futexes
 * @vs:		The list of futexes to wait on
 * @count:	The number of objects
 * @to:		Timeout before giving up and returning to userspace
 *
 * Entry point for the FUTEX_WAIT_MULTIPLE futex operation, this function
 * sleeps on a group of futexes and returns on the first futex that is
 * wake, or after the timeout has elapsed.
 *
 * Return:
 *  - >=0 - Hint to the futex that was awoken
 *  - <0  - On error
 */
int futex_wait_multiple(struct futex_vector *vs, unsigned int count,
			struct hrtimer_sleeper *to)
{
	int ret, hint = 0;

	if (to)
		hrtimer_sleeper_start_expires(to, HRTIMER_MODE_ABS);

	while (1) {
		ret = futex_wait_multiple_setup(vs, count, &hint);
		if (ret) {
			if (ret > 0) {
				/* A futex was woken during setup */
				ret = hint;
			}
			return ret;
		}

		futex_sleep_multiple(vs, count, to);

		__set_current_state(TASK_RUNNING);

		ret = unqueue_multiple(vs, count);
		if (ret >= 0)
			return ret;

		if (to && !to->task)
			return -ETIMEDOUT;
		else if (signal_pending(current))
			return -ERESTARTSYS;
		/*
		 * The final case is a spurious wakeup, for
		 * which just retry.
		 */
	}
}

/**
 * futex_wait_setup() - Prepare to wait on a futex
 * @uaddr:	the futex userspace address
 * @val:	the expected value
 * @flags:	futex flags (FLAGS_SHARED, etc.)
 * @q:		the associated futex_q
 * @hb:		storage for hash_bucket pointer to be returned to caller
 *
 * Setup the futex_q and locate the hash_bucket.  Get the futex value and
 * compare it with the expected value.  Handle atomic faults internally.
 * Return with the hb lock held on success, and unlocked on failure.
 *
 * Return:
 *  -  0 - uaddr contains val and hb has been locked;
 *  - <1 - -EFAULT or -EWOULDBLOCK (uaddr does not contain val) and hb is unlocked
 */

/*根据uaddr flags 设置hash的key值,并判断uaddr是否为预期值*/
int futex_wait_setup(u32 __user *uaddr, u32 val, unsigned int flags,
		     struct futex_q *q, struct futex_hash_bucket **hb)
{
	u32 uval;
	int ret;

	/*
	 * Access the page AFTER the hash-bucket is locked.
	 * Order is important:
	 *
	 *   Userspace waiter: val = var; if (cond(val)) futex_wait(&var, val);
	 *   Userspace waker:  if (cond(var)) { var = new; futex_wake(&var); }
	 *
	 * The basic logical guarantee of a futex is that it blocks ONLY
	 * if cond(var) is known to be true at the time of blocking, for
	 * any cond.  If we locked the hash-bucket after testing *uaddr, that
	 * would open a race condition where we could block indefinitely with
	 * cond(var) false, which would violate the guarantee.
	 *
	 * On the other hand, we insert q and release the hash-bucket only
	 * after testing *uaddr.  This guarantees that futex_wait() will NOT
	 * absorb a wakeup if *uaddr does not match the desired values
	 * while the syscall executes.
	 */
retry:

	ret = get_futex_key(uaddr, flags & FLAGS_SHARED, &q->key, FUTEX_READ);//
	if (unlikely(ret != 0))
		return ret;

retry_private:
	*hb = futex_q_lock(q);

	ret = futex_get_value_locked(&uval, uaddr);

	if (ret) {
		futex_q_unlock(*hb);

		ret = get_user(uval, uaddr);
		if (ret)
			return ret;

		if (!(flags & FLAGS_SHARED))
			goto retry_private;

		goto retry;
	}

	if (uval != val) {
		futex_q_unlock(*hb);
		ret = -EWOULDBLOCK;
	}

	return ret;
}
/*将线程放入等待队列,并使其阻塞*/
int futex_wait(u32 __user *uaddr, unsigned int flags, u32 val, ktime_t *abs_time, u32 bitset)
{
	struct hrtimer_sleeper timeout, *to;//定义一个高精度定时器（hrtimer）用于处理超时等待
	struct restart_block *restart;//处理信号重启机制的数据结构
	struct futex_hash_bucket *hb;//Futex 哈希桶，用于加锁和管理 Futex 等待队列
	struct futex_q q = futex_q_init;//初始化 Futex 等待队列中的条目
	int ret;

	if (!bitset)
		return -EINVAL;
	q.bitset = bitset;
	/*设置futex的定时器,如果 abs_time 有值，则根据超时时间初始化 hrtimer*/
	to = futex_setup_timer(abs_time, &timeout, flags,
			       current->timer_slack_ns);
retry:
	/*
	 * Prepare to wait on uaddr. On success, it holds hb->lock and q
	 * is initialized.
	 */
	/**/
	ret = futex_wait_setup(uaddr, val, flags, &q, &hb);//获取hash key 并确定uaddr为预期值;
	if (ret)
		goto out;

	/*futex_wait_queue
	 *1.将当前线程加入futex等待队列,
	 *2.启动定时器，
	 *3.重新调度
	 */
	futex_wait_queue(hb, &q, to);
/****************************************************************/
/*                      线程被阻塞,等待被唤醒                     */
/****************************************************************/

	/* 线程被唤醒，并成功从等待队列移除 */
	ret = 0;
	if (!futex_unqueue(&q))
		goto out;
	ret = -ETIMEDOUT;
	if (to && !to->task)//如果定时器超时并且没有任务需要继续处理，跳转到 out 退出
		goto out;

	/*
	 * We expect signal_pending(current), but we might be the
	 * victim of a spurious wakeup as well.
	 */
	if (!signal_pending(current))
		goto retry;

	ret = -ERESTARTSYS;
	if (!abs_time)
		goto out;
	/*设置 restart_block 以便处理信号中断的重启机制*/
	restart = &current->restart_block;
	restart->futex.uaddr = uaddr;
	restart->futex.val = val;
	restart->futex.time = *abs_time;
	restart->futex.bitset = bitset;
	restart->futex.flags = flags | FLAGS_HAS_TIMEOUT;

	/*设置重启函数为 futex_wait_restart*/
	ret = set_restart_fn(restart, futex_wait_restart);
out:
	/*如果定时器存在，取消定时器并销毁 hrtimer*/
	if (to) {
		hrtimer_cancel(&to->timer);
		destroy_hrtimer_on_stack(&to->timer);
	}
	return ret;
}

static long futex_wait_restart(struct restart_block *restart)
{
	u32 __user *uaddr = restart->futex.uaddr;
	ktime_t t, *tp = NULL;

	if (restart->futex.flags & FLAGS_HAS_TIMEOUT) {
		t = restart->futex.time;
		tp = &t;
	}
	restart->fn = do_no_restart_syscall;

	return (long)futex_wait(uaddr, restart->futex.flags,
				restart->futex.val, tp, restart->futex.bitset);
}

