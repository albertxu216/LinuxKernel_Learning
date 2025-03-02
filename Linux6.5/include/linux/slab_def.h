/* SPDX-License-Identifier: GPL-2.0 */
#ifndef _LINUX_SLAB_DEF_H
#define	_LINUX_SLAB_DEF_H

#include <linux/kfence.h>
#include <linux/reciprocal_div.h>

/*
 * Definitions unique to the original Linux SLAB allocator.
 */
/*每个slab缓存都由 kmem_cache 数据结构表示*/
struct kmem_cache {
	/*1.每个CPU都有一个，表示本地对象缓冲池*/
	struct array_cache __percpu *cpu_cache;

/* 1) Cache tunables. Protected by slab_mutex */

	/*表示在当前 CPU 的本地对象缓冲池 array_cache 为空时，
	 *从共享对象缓冲池或 slabs_partial/slabs_free 列表中迁移空闲对象的数目
	 */
	unsigned int batchcount;

	/*当本地对象缓冲池中的空闲对象的数目大于 limit 时，
	 *会主动释放 batchcount 个对象，便于内核回收和销毁 slab*/
	unsigned int limit;

	/**/
	unsigned int shared;

	/*对象的长度，这个长度要加上 align 对齐字节*/
	unsigned int size;
	struct reciprocal_value reciprocal_buffer_size;
/* 2) touched by every alloc & free from the backend */
	/*对象的分配掩码*/
	slab_flags_t flags;		/* constant flags */
	/*一个 slab 中最多有多少个对象*/
	unsigned int num;		/* # of objs per slab */

/* 3) cache_grow/shrink */
	/* order of pgs per slab (2^n) */
	/*一个slab分配器从伙伴系统申请多少个物理页面2^gfporder个*/
	unsigned int gfporder;

	/* force GFP flags, e.g. GFP_DMA */
	gfp_t allocflags;
	/*一个 slab 中可以有多少个不同的缓存行*/
	size_t colour;			/* cache colouring range */
	/*着色区的长度，和 L1 缓存行大小相同*/
	unsigned int colour_off;	/* colour offset */
	/*每个对象要占用 1 字节来存放 freelist*/
	unsigned int freelist_size;

	/* constructor func */
	/*构造函数*/
	void (*ctor)(void *obj);

/* 4) cache creation/removal */
	/*slab 描述符的名称*/
	const char *name;

	/*在slab_caches 链表中，该slab缓存用于找到前一个和后一个slab缓存*/
	struct list_head list;

	/*当前slab缓存被引用的次数，
	 *当创建其他slab描述符并引用该slab描述符时，
	 *会增加引用次数
	 */
	int refcount;

	/*对象的实际大小*/
	int object_size;

	/*对齐的长度*/
	int align;

/* 5) statistics */
#ifdef CONFIG_DEBUG_SLAB
	unsigned long num_active;
	unsigned long num_allocations;
	unsigned long high_mark;
	unsigned long grown;
	unsigned long reaped;
	unsigned long errors;
	unsigned long max_freeable;
	unsigned long node_allocs;
	unsigned long node_frees;
	unsigned long node_overflow;
	atomic_t allochit;
	atomic_t allocmiss;
	atomic_t freehit;
	atomic_t freemiss;

	/*
	 * If debugging is enabled, then the allocator can add additional
	 * fields and/or padding to every object. 'size' contains the total
	 * object size including these internal fields, while 'obj_offset'
	 * and 'object_size' contain the offset to the user object and its
	 * size.
	 */
	int obj_offset;
#endif /* CONFIG_DEBUG_SLAB */

#ifdef CONFIG_KASAN_GENERIC
	struct kasan_cache kasan_info;
#endif

#ifdef CONFIG_SLAB_FREELIST_RANDOM
	unsigned int *random_seq;
#endif

#ifdef CONFIG_HARDENED_USERCOPY
	unsigned int useroffset;	/* Usercopy region offset */
	unsigned int usersize;		/* Usercopy region size */
#endif
	/*NUMA系统中，每个节点下对应一个slab节点*/
	struct kmem_cache_node *node[MAX_NUMNODES];
};

static inline void *nearest_obj(struct kmem_cache *cache, const struct slab *slab,
				void *x)
{
	void *object = x - (x - slab->s_mem) % cache->size;
	void *last_object = slab->s_mem + (cache->num - 1) * cache->size;

	if (unlikely(object > last_object))
		return last_object;
	else
		return object;
}

/*
 * We want to avoid an expensive divide : (offset / cache->size)
 *   Using the fact that size is a constant for a particular cache,
 *   we can replace (offset / cache->size) by
 *   reciprocal_divide(offset, cache->reciprocal_buffer_size)
 */
static inline unsigned int obj_to_index(const struct kmem_cache *cache,
					const struct slab *slab, void *obj)
{
	u32 offset = (obj - slab->s_mem);
	return reciprocal_divide(offset, cache->reciprocal_buffer_size);
}

static inline int objs_per_slab(const struct kmem_cache *cache,
				     const struct slab *slab)
{
	if (is_kfence_address(slab_address(slab)))
		return 1;
	return cache->num;
}

#endif	/* _LINUX_SLAB_DEF_H */
