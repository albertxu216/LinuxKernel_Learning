// SPDX-License-Identifier: GPL-2.0
/*
 * vgic init sequence tests
 *
 * Copyright (C) 2020, Red Hat, Inc.
 */
#include <linux/kernel.h>
#include <sys/syscall.h>
#include <asm/kvm.h>
#include <asm/kvm_para.h>

#include "test_util.h"
#include "kvm_util.h"
#include "processor.h"
#include "vgic.h"

#define NR_VCPUS		4

#define REG_OFFSET(vcpu, offset) (((uint64_t)vcpu << 32) | offset)

#define GICR_TYPER 0x8

#define VGIC_DEV_IS_V2(_d) ((_d) == KVM_DEV_TYPE_ARM_VGIC_V2)
#define VGIC_DEV_IS_V3(_d) ((_d) == KVM_DEV_TYPE_ARM_VGIC_V3)

struct vm_gic {
	struct kvm_vm *vm;//kvm虚拟机
	int gic_fd;//
	uint32_t gic_dev_type;//设备类型
};

static uint64_t max_phys_size;

/*
 * Helpers to access a redistributor register and verify the ioctl() failed or
 * succeeded as expected, and provided the correct value on success.
 */
static void v3_redist_reg_get_errno(int gicv3_fd, int vcpu, int offset,
				    int want, const char *msg)
{
	uint32_t ignored_val;
	int ret = __kvm_device_attr_get(gicv3_fd, KVM_DEV_ARM_VGIC_GRP_REDIST_REGS,
					REG_OFFSET(vcpu, offset), &ignored_val);

	TEST_ASSERT(ret && errno == want, "%s; want errno = %d", msg, want);
}

static void v3_redist_reg_get(int gicv3_fd, int vcpu, int offset, uint32_t want,
			      const char *msg)
{
	uint32_t val;

	kvm_device_attr_get(gicv3_fd, KVM_DEV_ARM_VGIC_GRP_REDIST_REGS,
			    REG_OFFSET(vcpu, offset), &val);
	TEST_ASSERT(val == want, "%s; want '0x%x', got '0x%x'", msg, want, val);
}

/* dummy guest code */
static void guest_code(void)
{
	GUEST_SYNC(0);
	GUEST_SYNC(1);
	GUEST_SYNC(2);
	GUEST_DONE();
}

/* we don't want to assert on run execution, hence that helper */
static int run_vcpu(struct kvm_vcpu *vcpu)
{
	/*公共函数，riscv可用*/
	return __vcpu_run(vcpu) ? -errno : 0;
}

static struct vm_gic vm_gic_create_with_vcpus(uint32_t gic_dev_type,
					      uint32_t nr_vcpus,
					      struct kvm_vcpu *vcpus[])
{
	struct vm_gic v;

	v.gic_dev_type = gic_dev_type;
	/*公共函数，riscv可以直接用*/
	v.vm = vm_create_with_vcpus(nr_vcpus, guest_code, vcpus);
	v.gic_fd = kvm_create_device(v.vm, gic_dev_type);

	return v;
}

static struct vm_gic vm_gic_create_barebones(uint32_t gic_dev_type)
{
	struct vm_gic v;

	v.gic_dev_type = gic_dev_type;
	v.vm = vm_create_barebones();
	v.gic_fd = kvm_create_device(v.vm, gic_dev_type);

	return v;
}

/*公共函数，riscv可用*/
static void vm_gic_destroy(struct vm_gic *v)
{
	close(v->gic_fd);
	kvm_vm_free(v->vm);
}

struct vgic_region_attr {
	uint64_t attr;
	uint64_t size;
	uint64_t alignment;
};

struct vgic_region_attr gic_v3_dist_region = {
	.attr = KVM_VGIC_V3_ADDR_TYPE_DIST,
	.size = 0x10000,
	.alignment = 0x10000,
};

struct vgic_region_attr gic_v3_redist_region = {
	.attr = KVM_VGIC_V3_ADDR_TYPE_REDIST,
	.size = NR_VCPUS * 0x20000,
	.alignment = 0x10000,
};

struct vgic_region_attr gic_v2_dist_region = {
	.attr = KVM_VGIC_V2_ADDR_TYPE_DIST,
	.size = 0x1000,
	.alignment = 0x1000,
};

struct vgic_region_attr gic_v2_cpu_region = {
	.attr = KVM_VGIC_V2_ADDR_TYPE_CPU,
	.size = 0x2000,
	.alignment = 0x1000,
};

/**
 * Helper routine that performs KVM device tests in general. Eventually the
 * ARM_VGIC (GICv2 or GICv3) device gets created with an overlapping
 * DIST/REDIST (or DIST/CPUIF for GICv2). Assumption is 4 vcpus are going to be
 * used hence the overlap. In the case of GICv3, A RDIST region is set at @0x0
 * and a DIST region is set @0x70000. The GICv2 case sets a CPUIF @0x0 and a
 * DIST region @0x1000.
 */
/*公共函数,riscv可用*/
/*验证 KVM在处理 ARM VGIC 设备地址配置时的行为
 *该函数通过设置分发器和重分发器的地址，
 *测试 KVM 在各种异常情况下的错误处理能力，确保虚拟化环境的正确性和稳定性。
 */
static void subtest_dist_rdist(struct vm_gic *v)
{
	int ret;
	uint64_t addr;
	struct vgic_region_attr rdist; /* vgic的重发器*/
	struct vgic_region_attr dist;/* GICv3中的分发器 */
	/* 根据GIC设备类型（v2或v3）设置DIST和REDIST区域的属性*/
	rdist = VGIC_DEV_IS_V3(v->gic_dev_type) ? gic_v3_redist_region
						: gic_v2_cpu_region;
	dist = VGIC_DEV_IS_V3(v->gic_dev_type) ? gic_v3_dist_region
						: gic_v2_dist_region;

	/* Check existing group/attributes */
	/*公共函数,riscv可用 
	 *1.确认 DIST 和 REDIST/ 的地址属性是否存在。
	 */
	kvm_has_device_attr(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR, dist.attr);
	kvm_has_device_attr(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR, rdist.attr);

	/* 公共函数,riscv可用 
	 *2. 使用无效属性值-1，测试不存在的属性，
	 *   检查不存在的属性（期望错误返回）
	 */
	ret = __kvm_has_device_attr(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR, -1);
	TEST_ASSERT(ret && errno == ENXIO, "attribute not supported");

	/* misaligned DIST and REDIST address settings */
	/*
	 *3. 验证 KVM 是否拒绝未对齐的地址设置
	 *   将地址设置为小于对齐要求的值（alignment / 0x10）
	 *   调用 __kvm_device_attr_set 设置 DIST 和 REDIST/CPUIF 的基地址。
	 */
	addr = dist.alignment / 0x10;
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    dist.attr, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "GIC dist base not aligned");

	addr = rdist.alignment / 0x10;
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    rdist.attr, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "GIC redist/cpu base not aligned");

	/* out of range address */
	/*
	 *4. 测试物理地址超出范围的情况,kvm是否能检测出来 
	 */
	addr = max_phys_size;
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    dist.attr, &addr);
	TEST_ASSERT(ret && errno == E2BIG, "dist address beyond IPA limit");

	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    rdist.attr, &addr);
	TEST_ASSERT(ret && errno == E2BIG, "redist address beyond IPA limit");

	/* Space for half a rdist (a rdist is: 2 * rdist.alignment). */
	/*设置 REDIST 基地址为 max_phys_size - dist.alignment，使其部分区域超出 IPA，同样期望 E2BIG。*/
	addr = max_phys_size - dist.alignment;
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    rdist.attr, &addr);
	TEST_ASSERT(ret && errno == E2BIG,
			"half of the redist is beyond IPA limit");

	/* set REDIST base address @0x0*/
	/*设置rdist*/
	addr = 0x00000;
	kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    rdist.attr, &addr);

	/* Attempt to create a second legacy redistributor region */
	addr = 0xE0000;
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    rdist.attr, &addr);
	TEST_ASSERT(ret && errno == EEXIST, "GIC redist base set again");

	/*设置vgic3 的混合区域配置 aia是否需要对这部分进行测试?
	 *检查是否允许混合使用传统 REDIST 和新 REDIST_REGION（GICv3 特性）
	 */
	ret = __kvm_has_device_attr(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				     KVM_VGIC_V3_ADDR_TYPE_REDIST);
	if (!ret) {
		/* Attempt to mix legacy and new redistributor regions */
		/*如果支持混合使用传统REDIST 和新 REDIST_REGION,则配置新的REDIST_REGION*/
		addr = REDIST_REGION_ATTR_ADDR(NR_VCPUS, 0x100000, 0, 0);
		ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
					    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
		TEST_ASSERT(ret && errno == EINVAL,
			    "attempt to mix GICv3 REDIST and REDIST_REGION");
	}

	/*
	 * Set overlapping DIST / REDIST, cannot be detected here. Will be detected
	 * on first vcpu run instead.
	 */
	/*将将 DIST 的地址设置为与 REDIST重叠*/
	addr = rdist.size - rdist.alignment;
	kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    dist.attr, &addr);
}

/* Test the new REDIST region API */
/*测试 ARM GICv3 中新的重分发器（Redistributor, REDIST）区域 API，
 *确保 KVM 能够正确处理重分发器区域的配置，并检测各种错误情况
 */
static void subtest_v3_redist_regions(struct vm_gic *v)
{
	uint64_t addr, expected_addr;
	int ret;

	/*1. 检查 KVM 设备是否支持 REDIST 地址类型属性
	 *   验证 KVM 是否支持多重分发器区域功能。
	 */
	ret = __kvm_has_device_attr(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST);
	TEST_ASSERT(!ret, "Multiple redist regions advertised");

	/*2. 测试设置带有非零标志位的重分发器区域，期望返回 EINVAL
	 *   测试无效的重分发器区域配置
	 *   应当为:REDIST_REGION_ATTR_ADDR(NR_VCPUS, 0x100000, 0, 0)
	 */
	addr = REDIST_REGION_ATTR_ADDR(NR_VCPUS, 0x100000, 2, 0);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "redist region attr value with flags != 0");

	/*3. 尝试设置带有 count 为 0 的 redistributor 区域属性（无效）
	 */
	addr = REDIST_REGION_ATTR_ADDR(0, 0x100000, 0, 0);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "redist region attr value with count== 0");

	/*4. 尝试用非零的索引注册第一个 redistributor 区域（无效）
	 *   正确:addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 0);
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 1);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL,
		    "attempt to register the first rdist region with index != 0");

	/*5. 尝试设置 未对齐地址的 redistributor 区域（无效）
	 *   应当是:REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 1);
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x201000, 0, 1);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "rdist region with misaligned address");

	/*6. 成功设置有效的 redistributor 区域
	 *   REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 0);
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 0);
	kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	/*7. 重复索引
	 *   尝试注册已经使用过的索引（无效）
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 1);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "register an rdist region with already used index");

	/*8. 重叠区域注册
	 *   地址与现有区域（0x200000）重叠
	 */
	addr = REDIST_REGION_ATTR_ADDR(1, 0x210000, 0, 2);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL,
		    "register an rdist region overlapping with another one");

	/*9. 尝试注册索引不按 +1 增加的 redistributor 区域,即不连续
	 *   正确:REDIST_REGION_ATTR_ADDR(1, 0x240000, 0, 1)
	 */
	addr = REDIST_REGION_ATTR_ADDR(1, 0x240000, 0, 2);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "register redist region with index not +1");

	/*10. 成功设置第二个 redistributor 区域，索引正确*/
	addr = REDIST_REGION_ATTR_ADDR(1, 0x240000, 0, 1);
	kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	/*11. 尝试注册超出 IPA 范围的基础地址的 redistributor 区域（无效） */
	addr = REDIST_REGION_ATTR_ADDR(1, max_phys_size, 0, 2);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == E2BIG,
		    "register redist region with base address beyond IPA range");
	
	/*12. 尝试注册超出 IPA 范围的结束地址的 redistributor 区域（无效）*/
	/* The last redist is above the pa range. */
	addr = REDIST_REGION_ATTR_ADDR(2, max_phys_size - 0x30000, 0, 2);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == E2BIG,
		    "register redist region with top address beyond IPA range");

	/*13. 尝试混合使用 REDIST 类型和 REDIST_REGION（无效）*/
	addr = 0x260000;
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST, &addr);
	TEST_ASSERT(ret && errno == EINVAL,
		    "Mix KVM_VGIC_V3_ADDR_TYPE_REDIST and REDIST_REGION");

    // 现在有 2 个 redistributor 区域：
    // 区域 0：0x200000 地址，包含 2 个 redistributor
    // 区域 1：0x240000 地址，包含 1 个 redistributor

	/*14. 尝试读取两个redistributor 区域的属性*/
	/*14.1 区域一:*/
	addr = REDIST_REGION_ATTR_ADDR(0, 0, 0, 0);
	expected_addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 0);
	ret = __kvm_device_attr_get(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(!ret && addr == expected_addr, "read characteristics of region #0");
	/*14.2 区域二:*/
	addr = REDIST_REGION_ATTR_ADDR(0, 0, 0, 1);
	expected_addr = REDIST_REGION_ATTR_ADDR(1, 0x240000, 0, 1);
	ret = __kvm_device_attr_get(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(!ret && addr == expected_addr, "read characteristics of region #1");
	
	/*15. 读取一个不存在得区域*/
	addr = REDIST_REGION_ATTR_ADDR(0, 0, 0, 2);
	ret = __kvm_device_attr_get(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == ENOENT, "read characteristics of non existing region");

	/*16. 设置分发器distributor 地址*/
	addr = 0x260000;
	kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_DIST, &addr);

	/*17. 尝试注册一个和distributor 地址冲突的redist*/
	addr = REDIST_REGION_ATTR_ADDR(1, 0x260000, 0, 2);
	ret = __kvm_device_attr_set(v->gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "register redist region colliding with dist");
}

/*
 * VGIC KVM device is created and initialized before the secondary CPUs
 * get created
 */
static void test_vgic_then_vcpus(uint32_t gic_dev_type)
{
	struct kvm_vcpu *vcpus[NR_VCPUS];
	struct vm_gic v;
	int ret, i;

	/*公共函数,riscv可用
	 *1. 启用一个虚拟机
	 */
	v = vm_gic_create_with_vcpus(gic_dev_type, 1, vcpus);

	/*公共函数,riscv可用
	 *2. 检查dist rdist区域
	 */
	subtest_dist_rdist(&v);

	/* Add the rest of the VCPUs */
	for (i = 1; i < NR_VCPUS; ++i)
		/*公共函数,riscv可用*/
		vcpus[i] = vm_vcpu_add(v.vm, i, guest_code);

	ret = run_vcpu(vcpus[3]);
	TEST_ASSERT(ret == -EINVAL, "dist/rdist overlap detected on 1st vcpu run");

	vm_gic_destroy(&v);
}

/* All the VCPUs are created before the VGIC KVM device gets initialized */
/* 所有vcpu均被创建在vgic之前，先创建4个vcpu，再初始化vgic
 * 设置重叠的distributor和redistributor地址，运行VCPU时检测错误
 * 预期结果：运行VCPU时返回-EINVAL，因为地址重叠被检测到。
 * 
 * 这个函数测试了在 ARM 架构下，当 GIC 的 distributor 和 redistributor（或 CPU 接口）地址重叠时，
 * KVM 是否能正确检测到这种错误配置，并在运行 VCPU 时返回 -EINVAL。
 */
static void test_vcpus_then_vgic(uint32_t gic_dev_type)
{
	struct kvm_vcpu *vcpus[NR_VCPUS];
	struct vm_gic v;
	int ret;

	/*公共函数,riscv可用
	 *1. 创建一个虚拟机，
	 *   并且为该虚拟机初始化 NR_VCPUS 个虚拟 CPU，同时配置一个 GIC设备
	 */
	v = vm_gic_create_with_vcpus(gic_dev_type, NR_VCPUS, vcpus);

	/*公共函数,riscv可用
	 *2. 设置 GIC 的地址并制造重叠
	 *   设置 GIC 的 distributor（分发器）和 redistributor（重分发器，对于 GICv3）的内存映射地址，并且故意让这些地址范围重叠。
	 */
	subtest_dist_rdist(&v);

	/*公共函数,riscv可用
	 *3. 运行vcpu 并验证错误
	 *   尝试验证第四个vcpu，并验证kvm是否能检测到问题并返回正确错误码；
	 */
	ret = run_vcpu(vcpus[3]);
	TEST_ASSERT(ret == -EINVAL, "dist/rdist overlap detected on 1st vcpu run");

	/*4. 销毁虚拟机和 GIC 设备，释放相关资源*/
	vm_gic_destroy(&v);
}

#define KVM_VGIC_V2_ATTR(offset, cpu) \
	(FIELD_PREP(KVM_DEV_ARM_VGIC_OFFSET_MASK, offset) | \
	 FIELD_PREP(KVM_DEV_ARM_VGIC_CPUID_MASK, cpu))

#define GIC_CPU_CTRL	0x00

/*vgic2 暂不做研究*/
static void test_v2_uaccess_cpuif_no_vcpus(void)
{
	struct vm_gic v;
	u64 val = 0;
	int ret;

	v = vm_gic_create_barebones(KVM_DEV_TYPE_ARM_VGIC_V2);
	subtest_dist_rdist(&v);

	ret = __kvm_has_device_attr(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CPU_REGS,
				    KVM_VGIC_V2_ATTR(GIC_CPU_CTRL, 0));
	TEST_ASSERT(ret && errno == EINVAL,
		    "accessed non-existent CPU interface, want errno: %i",
		    EINVAL);
	ret = __kvm_device_attr_get(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CPU_REGS,
				    KVM_VGIC_V2_ATTR(GIC_CPU_CTRL, 0), &val);
	TEST_ASSERT(ret && errno == EINVAL,
		    "accessed non-existent CPU interface, want errno: %i",
		    EINVAL);
	ret = __kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CPU_REGS,
				    KVM_VGIC_V2_ATTR(GIC_CPU_CTRL, 0), &val);
	TEST_ASSERT(ret && errno == EINVAL,
		    "accessed non-existent CPU interface, want errno: %i",
		    EINVAL);

	vm_gic_destroy(&v);
}

/*测试了 GICv3（Generic Interrupt Controller v3）中 Redistributor区域（REDIST）相关的功能，
 *具体包括验证中断重定向区域的创建、初始化、地址配置等问题。该测试步骤包括三个主要部分，
 *验证了不同的配置、错误处理和初始化步骤。
 */
static void test_v3_new_redist_regions(void)
{
	struct kvm_vcpu *vcpus[NR_VCPUS];
	void *dummy = NULL;
	struct vm_gic v;// 定义一个虚拟机的 GIC
	uint64_t addr;
	int ret;

	/*公共函数，riscv可用
	 *1. 启用一个虚拟机和vcpu
	 *   调用 subtest_v3_redist_regions 测试 REDIST 区域相关的设置
	 *   初始化 VGIC 控制器
	 *   尝试运行第三个虚拟 CPU，并检查是否没有足够的 REDIST 区域
	 *   运行 VCPU，预期返回 -ENXIO，因为重分发器数量不足以覆盖所有 VCPUs（假设 NR_VCPUS > 3）
	 */
	v = vm_gic_create_with_vcpus(KVM_DEV_TYPE_ARM_VGIC_V3, NR_VCPUS, vcpus);

	subtest_v3_redist_regions(&v);

	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,
			    KVM_DEV_ARM_VGIC_CTRL_INIT, NULL);

	ret = run_vcpu(vcpus[3]);
	TEST_ASSERT(ret == -ENXIO, "running without sufficient number of rdists");
	vm_gic_destroy(&v);

	/* step2 */
	/*重新创建虚拟机和 VCPUs
	 *调用 subtest_v3_redist_regions 测试 REDIST 区域相关的设置
	 *添加第三个区域：addr = REDIST_REGION_ATTR_ADDR(1, 0x280000, 0, 2)
	 *未初始化 VGIC，直接运行 VCPU,测试是否能检测出vgic未初始化的情况;
	 */
	v = vm_gic_create_with_vcpus(KVM_DEV_TYPE_ARM_VGIC_V3, NR_VCPUS, vcpus);
	subtest_v3_redist_regions(&v);

	addr = REDIST_REGION_ATTR_ADDR(1, 0x280000, 0, 2);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	ret = run_vcpu(vcpus[3]);
	TEST_ASSERT(ret == -EBUSY, "running without vgic explicit init");

	vm_gic_destroy(&v);

	/* step 3 */

	/*重新创建虚拟机和 VCPUs
	 *调用 subtest_v3_redist_regions 测试 REDIST 区域相关的设置
	 *使用无效指针（dummy = NULL）设置区域 
	 *等待报错;
	 *创建一个有效的区域;并初始化vgic
	 *尝试成功运行一个vcpu
	 */
	v = vm_gic_create_with_vcpus(KVM_DEV_TYPE_ARM_VGIC_V3, NR_VCPUS, vcpus);
	subtest_v3_redist_regions(&v);

	ret = __kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, dummy);
	TEST_ASSERT(ret && errno == EFAULT,
		    "register a third region allowing to cover the 4 vcpus");
	/*设置一个有效的区域并创建一个*/
	addr = REDIST_REGION_ATTR_ADDR(1, 0x280000, 0, 2);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,
			    KVM_DEV_ARM_VGIC_CTRL_INIT, NULL);

	ret = run_vcpu(vcpus[3]);
	TEST_ASSERT(!ret, "vcpu run");

	vm_gic_destroy(&v);
}

/*测试了 ARM64 KVM vgic3 中 GICR_TYPER 寄存器的访问，
 *涵盖了不同状态下的错误处理和正确性验证
 */
static void test_v3_typer_accesses(void)
{
	struct vm_gic v;
	uint64_t addr;
	int ret, i;

	/*1. 创建一个虚拟机，包含 NR_VCPUS 个虚拟 CPU*/
	v.vm = vm_create(NR_VCPUS);
	(void)vm_vcpu_add(v.vm, 0, guest_code);		//添加 vCPU 0，并加载访客代码

	/*2. 为虚拟机创建 vgic3 设备*/
	v.gic_fd = kvm_create_device(v.vm, KVM_DEV_TYPE_ARM_VGIC_V3);

	(void)vm_vcpu_add(v.vm, 3, guest_code);		//添加 vCPU 3，并加载访客代码

	/*3. 测试1：
	 *   尝试读取未创建的 vCPU 1 的 GICR_TYPER 寄存器，
	 *   预期返回 EINVAL
	 */
	v3_redist_reg_get_errno(v.gic_fd, 1, GICR_TYPER, EINVAL,
				"attempting to read GICR_TYPER of non created vcpu");

	(void)vm_vcpu_add(v.vm, 1, guest_code);		//添加 vCPU 1，并加载访客代码

	/*4. 测试2：
	 *   读取 vCPU 1 的 GICR_TYPER 寄存器，但 GIC 未初始化，
	 *   预期返回 EBUSY
	 */
	v3_redist_reg_get_errno(v.gic_fd, 1, GICR_TYPER, EBUSY,
				"read GICR_TYPER before GIC initialized");

	(void)vm_vcpu_add(v.vm, 2, guest_code);		//添加 vCPU 2，并加载访客代码
	
	/*5. 初始化 GIC*/
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,
			    KVM_DEV_ARM_VGIC_CTRL_INIT, NULL);
	
	/*6. 测试3：
	 *   在设置 redist 区域之前，
	 *   读取每个 vCPU 的 GICR_TYPER 寄存器
	 */
	for (i = 0; i < NR_VCPUS ; i++) {
		v3_redist_reg_get(v.gic_fd, i, GICR_TYPER, i * 0x100,
				  "read GICR_TYPER before rdist region setting");
	}

	/*7. 设置 redist 区域的地址属性，
	 *   参数为 vCPU 2，
	 *   地址 0x200000，
	 *   标志位 0
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 0);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	/* The 2 first rdists should be put there (vcpu 0 and 3) */
	/*8. 测试4：
	 *   读取 vCPU 0 和 3 的 GICR_TYPER寄存器，
	 *   验证 redist 区域设置后正确性
	 *   目前已经创建了虚拟机、创建了4个vcpu并、
	 *   创建并初始化了vgic设备、设置了 redist 区域的地址属性；
	 *   所有条件均已满足，理应读取成功
	 */
	v3_redist_reg_get(v.gic_fd, 0, GICR_TYPER, 0x0, "read typer of rdist #0");
	v3_redist_reg_get(v.gic_fd, 3, GICR_TYPER, 0x310, "read typer of rdist #1");

	/*9. 测试5：
	 *   尝试设置另一个 redist 区域，与现有区域冲突，
	 *   预期返回 EINVAL
	 */
	addr = REDIST_REGION_ATTR_ADDR(10, 0x100000, 0, 1);
	ret = __kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	TEST_ASSERT(ret && errno == EINVAL, "collision with previous rdist region");

	/*10. 测试6：
	 *    读取 vCPU 1 和 vcpu 2 的 GICR_TYPER，未设置 redist 区域，验证行为
	 */
	v3_redist_reg_get(v.gic_fd, 1, GICR_TYPER, 0x100,
			  "no redist region attached to vcpu #1 yet, last cannot be returned");
	v3_redist_reg_get(v.gic_fd, 2, GICR_TYPER, 0x200,
			  "no redist region attached to vcpu #2, last cannot be returned");

	/*11. 设置另一个 redist 区域，参数为 vCPU 10，地址 0x20000，标志位 1*/
	addr = REDIST_REGION_ATTR_ADDR(10, 0x20000, 0, 1);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);
	
	/*12. 测试7：
	 *    读取 vCPU 1 和 2 的 GICR_TYPER，验证设置后正确性
	 */
	v3_redist_reg_get(v.gic_fd, 1, GICR_TYPER, 0x100, "read typer of rdist #1");
	v3_redist_reg_get(v.gic_fd, 2, GICR_TYPER, 0x210,
			  "read typer of rdist #1, last properly returned");
	
	/*13. 清理资源，销毁虚拟机和相关结构*/
	vm_gic_destroy(&v);
}

/*按照指定cpu id 顺序创建vcpu 035421*/
static struct vm_gic vm_gic_v3_create_with_vcpuids(int nr_vcpus,
						   uint32_t vcpuids[])
{
	struct vm_gic v;
	int i;

	/*1. 创建一个包含 nr_vcpus 个 vCPU 的虚拟机*/
	v.vm = vm_create(nr_vcpus);

	/*2. 按指定 ID 顺序添加每个 vCPU，并加载访客代码*/
	for (i = 0; i < nr_vcpus; i++)
		vm_vcpu_add(v.vm, vcpuids[i], guest_code);

	/*3. 为虚拟机创建 vgic3 设备*/
	v.gic_fd = kvm_create_device(v.vm, KVM_DEV_TYPE_ARM_VGIC_V3);

	return v;
}

/**
 * 测试 GICR_TYPER 的最后位与新的 redist 区域
 * redist 区域 #1 和 #2 是连续的
 * redist 区域 #0 @0x100000，容量 2 个 rdist
 *     rdist: 0, 3 (最后) 0x100000 ~ 0x140000
 * redist 区域 #1 @0x240000，容量 2 个 rdist
 *     rdist: 5, 4 (最后) 0x240000 ~ 0x280000
 * redist 区域 #2 @0x200000，容量 2 个 rdist
 *     rdist: 1, 2 		 0x200000 ~ 0x240000
 */

/*测试 GICv3 中 Redistributor 区域（REDIST REGION）的函数。
 *测试验证了 vCPU 被正确分配到 redist 区域，并且 GICR_TYPER 的值与 vCPU 的位置相关，
 *验证了多个 Redistributor 区域的配置、访问和正确性。
 *具体测试了 Redistributor 区域的初始化和在多个 Redistributor 区域间的中断映射。
 */
static void test_v3_last_bit_redist_regions(void)
{
	/*0. vCPU ID 顺序为 {0, 3, 5, 4, 1, 2}，总共 6 个 vCPU*/
	uint32_t vcpuids[] = { 0, 3, 5, 4, 1, 2 };
	struct vm_gic v;
	uint64_t addr;

	/*1. 创建一个带有指定 vCPU ID 的虚拟机和 vgic3 设备
	 *   vCPU ID 顺序为 {0, 3, 5, 4, 1, 2}，总共 6 个 vCPU
	 */
	v = vm_gic_v3_create_with_vcpuids(ARRAY_SIZE(vcpuids), vcpuids);

	/*2. 初始化GICv3控制器，执行中断控制器的初始化*/
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,
			    KVM_DEV_ARM_VGIC_CTRL_INIT, NULL);

	/*3. 设置第一个Redistributor区域的地址，
	 *   模拟一个2个Redistributor容量的区域，
	 *   起始地址0x100000 ，
	 *   索引0
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x100000, 0, 0);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	/*4. 设置第二个Redistributor区域的地址，
	 *   模拟一个2个Redistributor容量的区域，
	 *   起始地址0x240000
	 *   索引1
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x240000, 0, 1);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	/*5. 设置第三个Redistributor区域的地址，
	 *   模拟一个2个Redistributor容量的区域，
	 *   起始地址0x200000
	 *   索引2
	 */
	addr = REDIST_REGION_ATTR_ADDR(2, 0x200000, 0, 2);
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION, &addr);

	/*6. 读取并验证每个Redistributor区域的GICR_TYPER寄存器
	 *   确保不同的Redistributor区域能被正确地映射并返回预期的寄存器值
	 *
	 *   GICR_TYPER 寄存器的值与 vCPU 的分配和区域设置一致，
	 *   预期值可能反映基地址的低位部分或 vCPU ID 的某种编码
	 */
	v3_redist_reg_get(v.gic_fd, 0, GICR_TYPER, 0x000, "read typer of rdist #0");
	v3_redist_reg_get(v.gic_fd, 1, GICR_TYPER, 0x100, "read typer of rdist #1");
	v3_redist_reg_get(v.gic_fd, 2, GICR_TYPER, 0x200, "read typer of rdist #2");
	v3_redist_reg_get(v.gic_fd, 3, GICR_TYPER, 0x310, "read typer of rdist #3");
	v3_redist_reg_get(v.gic_fd, 5, GICR_TYPER, 0x500, "read typer of rdist #5");
	v3_redist_reg_get(v.gic_fd, 4, GICR_TYPER, 0x410, "read typer of rdist #4");

	vm_gic_destroy(&v);
}

/* Test last bit with legacy region */
/*测试了 ARM64 KVM vgic3 在单一 redist 区域下的 GICR_TYPER 寄存器值，
 *验证 vCPU ID 和“最后位”的设置；
 *它通过为每个虚拟CPU（vCPU）配置一个 Redistributor区域，
 *然后读取每个Redistributor的 GICR_TYPER寄存器，验证其值是否符合预期。
 */
static void test_v3_last_bit_single_rdist(void)
{
	uint32_t vcpuids[] = { 0, 3, 5, 4, 1, 2 };
	struct vm_gic v;
	uint64_t addr;
	/*1. 创建一个虚拟机,并按顺序创建vcpu 035412 */
	v = vm_gic_v3_create_with_vcpuids(ARRAY_SIZE(vcpuids), vcpuids);

	/*2. 创建并初始化vgic设备*/
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,
			    KVM_DEV_ARM_VGIC_CTRL_INIT, NULL);

	/*3. 设置单一 redist 区域的基地址为 0x10000*/
	addr = 0x10000;
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST, &addr);

	/*4. 验证每个 vCPU 的 GICR_TYPER 寄存器值，基于 vCPU ID 和是否为最后的vCPU 
	 *   vcpu2为最后一个添加的，其对应的redist区域应为最后的区域，应该+0x010
	 */
	v3_redist_reg_get(v.gic_fd, 0, GICR_TYPER, 0x000, "read typer of rdist #0");
	v3_redist_reg_get(v.gic_fd, 3, GICR_TYPER, 0x300, "read typer of rdist #1");
	v3_redist_reg_get(v.gic_fd, 5, GICR_TYPER, 0x500, "read typer of rdist #2");
	v3_redist_reg_get(v.gic_fd, 1, GICR_TYPER, 0x100, "read typer of rdist #3");
	//vCPU 2：预期值 0x210（vCPU ID 2 * 0x100 + 0x10，因其为最后一个）
	v3_redist_reg_get(v.gic_fd, 2, GICR_TYPER, 0x210, "read typer of rdist #3");
	
	vm_gic_destroy(&v);
}

/* Uses the legacy REDIST region API. */
/*检查 是否能识别到 redist 区域内存分配不充足的情况；
 *即vcpu与申请的redist区域大小不匹配
 */
static void test_v3_redist_ipa_range_check_at_vcpu_run(void)
{
	struct kvm_vcpu *vcpus[NR_VCPUS];
	struct vm_gic v;
	int ret, i;
	uint64_t addr;
	/*1. 初始创建一个虚拟机，包含一个 vCPU，并创建vgic设备*/
	v = vm_gic_create_with_vcpus(KVM_DEV_TYPE_ARM_VGIC_V3, 1, vcpus);

	/* Set space for 3 redists, we have 1 vcpu, so this succeeds. */
	/*2. 为 3 个 redist 区域分配空间，当前只有一个 vCPU，所以成功*/
	addr = max_phys_size - (3 * 2 * 0x10000);//预留三个redist区域，目前只有一个vcpu，所以空间富裕
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_REDIST, &addr);
	
	/*3. 设置 dist 区域地址为 0x00000，增加一个dist */
	addr = 0x00000;
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_V3_ADDR_TYPE_DIST, &addr);

	/* Add the rest of the VCPUs */
	/*4. 增加剩余的3个vcpu */
	for (i = 1; i < NR_VCPUS; ++i)
		vcpus[i] = vm_vcpu_add(v.vm, i, guest_code);

	/*5. 初始化vgic设备*/
	kvm_device_attr_set(v.gic_fd, KVM_DEV_ARM_VGIC_GRP_CTRL,
			    KVM_DEV_ARM_VGIC_CTRL_INIT, NULL);

	/* Attempt to run a vcpu without enough redist space. */
	/*6. 尝试运行 第三个 vcpu 
	 *   会运行失败，因为redist 区域可能不足以容纳所有 vCPU，
	 *   尽管 redist 区域为三个 vCPU 设置了空间，
	 *   但添加更多 vCPU 后，运行 vCPU 时会检测到内存不足
	 * 
	 *原因：redist 区域为 3 个 vCPU 设置空间，但添加了4个 vCPU，导致 vCPU 3 没有分配空间
	 *     运行 vCPU 2 时，KVM 检查所有 vCPU 的 redist 区域是否有效，发现不一致性
	 */
	ret = run_vcpu(vcpus[2]);
	TEST_ASSERT(ret && errno == EINVAL,
		"redist base+size above PA range detected on 1st vcpu run");

	/*7. 销毁虚拟机和相关资源*/
	vm_gic_destroy(&v);
}

/**/
static void test_v3_its_region(void)
{
	struct kvm_vcpu *vcpus[NR_VCPUS];
	struct vm_gic v;
	uint64_t addr;
	int its_fd, ret;
	/*1. 创建带有4个vcpu的虚拟机，并创建vgic设备*/
	v = vm_gic_create_with_vcpus(KVM_DEV_TYPE_ARM_VGIC_V3, NR_VCPUS, vcpus);
	/*2. 创建its设备*/
	its_fd = kvm_create_device(v.vm, KVM_DEV_TYPE_ARM_VGIC_ITS);

	/*3.测试ITS设备*/
	/*3.1 测试1：
	 *    设置不对其的地址（0x401000）*/
	addr = 0x401000;
	ret = __kvm_device_attr_set(its_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_ITS_ADDR_TYPE, &addr);
	TEST_ASSERT(ret && errno == EINVAL,
		"ITS region with misaligned address");

	/*3.2 测试2：
	 *    设置超出物理地址范围的地址 max_phys_size
	 */
	addr = max_phys_size;
	ret = __kvm_device_attr_set(its_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_ITS_ADDR_TYPE, &addr);
	TEST_ASSERT(ret && errno == E2BIG,
		"register ITS region with base address beyond IPA range");

	/*3.3 测试3：
	 *    设置部分超出范围的地址（max_phys_size - 0x10000）
	 */
	addr = max_phys_size - 0x10000;
	ret = __kvm_device_attr_set(its_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_ITS_ADDR_TYPE, &addr);
	TEST_ASSERT(ret && errno == E2BIG,
		"Half of ITS region is beyond IPA range");

	/* This one succeeds setting the ITS base */
	/*3.4 成功为ITS设置有效地址 0x400000*/
	addr = 0x400000;
	kvm_device_attr_set(its_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
			    KVM_VGIC_ITS_ADDR_TYPE, &addr);

	/*3.5 测试4：
	 *    为同一个ITS设备 重复设置 地址；
	 */
	addr = 0x300000;
	ret = __kvm_device_attr_set(its_fd, KVM_DEV_ARM_VGIC_GRP_ADDR,
				    KVM_VGIC_ITS_ADDR_TYPE, &addr);
	TEST_ASSERT(ret && errno == EEXIST, "ITS base set again");
	
	/*3.6 销毁ITS设备及虚拟机资源*/
	close(its_fd);
	vm_gic_destroy(&v);
}

/*
 * Returns 0 if it's possible to create GIC device of a given type (V2 or V3).
 */
/*测试vgic 设备的创建逻辑*/
int test_kvm_device(uint32_t gic_dev_type)
{
	struct kvm_vcpu *vcpus[NR_VCPUS];
	struct vm_gic v;
	uint32_t other;
	int ret;
	/*1. 创建 NR_VCPUS个 虚拟CPU*/
	v.vm = vm_create_with_vcpus(NR_VCPUS, guest_code, vcpus);

	/* try to create a non existing KVM device */
	/*2. 尝试创建一个不存在的设备类型，查看是否能检测出错误*/
	ret = __kvm_test_create_device(v.vm, 0);
	TEST_ASSERT(ret && errno == ENODEV, "unsupported device");

	/* trial mode */
	/*3. 尝试创建一个vgic2 或vgic3版本的虚拟设备*/
	ret = __kvm_test_create_device(v.vm, gic_dev_type);
	if (ret)
		return ret;
	v.gic_fd = kvm_create_device(v.vm, gic_dev_type);
	/*4. 再次创建一次相同类型的vgic设备，看是否能检测出错误*/
	ret = __kvm_create_device(v.vm, gic_dev_type);
	TEST_ASSERT(ret < 0 && errno == EEXIST, "create GIC device twice");

	/* try to create the other gic_dev_type */
	/*5. 再次尝试创建一个其他版本的vgic设备，看是否能检测出错误*/
	other = VGIC_DEV_IS_V2(gic_dev_type) ? KVM_DEV_TYPE_ARM_VGIC_V3
					     : KVM_DEV_TYPE_ARM_VGIC_V2;

	if (!__kvm_test_create_device(v.vm, other)) {
		ret = __kvm_create_device(v.vm, other);
		TEST_ASSERT(ret < 0 && (errno == EINVAL || errno == EEXIST),
				"create GIC device while other version exists");
	}

	vm_gic_destroy(&v);

	return 0;
}

void run_tests(uint32_t gic_dev_type)
{
	/*1. 通用测试：测试vcpu与vgic的初始化顺序会不会影响*/
	test_vcpus_then_vgic(gic_dev_type);//先创建cpu，再初始化vgic
	test_vgic_then_vcpus(gic_dev_type);//先创建vgic，在添加vcpu

	if (VGIC_DEV_IS_V2(gic_dev_type))
		test_v2_uaccess_cpuif_no_vcpus();

	if (VGIC_DEV_IS_V3(gic_dev_type)) {
		/*2. 测试v3模式中创建一个新的redistributor区域时,遇到错误时是否可以识别*/
		test_v3_new_redist_regions();
		/*3. 测试GICv3在不同错误情况下，是否可以正常访问typers*/
		test_v3_typer_accesses();
		/*4. 测试redistributor区域的最后一位*/
		test_v3_last_bit_redist_regions();
		/*5. 测试单个redistributor的最后一位*/
		test_v3_last_bit_single_rdist();
		/*6. 检查 redist 区域内存分配是否足够*/
		test_v3_redist_ipa_range_check_at_vcpu_run();
		/*7. 测试GICv3的ITS区域，aia下没有its相关功能的设备*/
		test_v3_its_region();
	}
}

/*主函数首先计算物理地址空间的最大尺寸（max_phys_size），
 *然后尝试检测系统对GICv3和GICv2的支持。
 *如果支持某种版本，就运行对应的测试集（run_tests）。
 *如果两种版本都不支持，则跳过测试。
 */
int main(int ac, char **av)
{
	int ret;
	int pa_bits;
	int cnt_impl = 0;

	pa_bits = vm_guest_mode_params[VM_MODE_DEFAULT].pa_bits;
	max_phys_size = 1ULL << pa_bits;//最高地址

	/*1. 对于vgic3，首先测试设备的创建逻辑，再运行测试用例*/
	ret = test_kvm_device(KVM_DEV_TYPE_ARM_VGIC_V3);
	if (!ret) {
		pr_info("Running GIC_v3 tests.\n");
		/*2. 运行测试用例*/
		run_tests(KVM_DEV_TYPE_ARM_VGIC_V3);
		cnt_impl++;
	}

	/*1. 对于vgic2，首先测试设备的创建逻辑，再运行测试用例*/
	ret = test_kvm_device(KVM_DEV_TYPE_ARM_VGIC_V2);
	if (!ret) {
		pr_info("Running GIC_v2 tests.\n");
		/*2. 运行测试用例*/
		run_tests(KVM_DEV_TYPE_ARM_VGIC_V2);
		cnt_impl++;
	}

	if (!cnt_impl) {
		print_skip("No GICv2 nor GICv3 support");
		exit(KSFT_SKIP);
	}
	return 0;
}
