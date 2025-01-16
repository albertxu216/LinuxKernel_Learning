// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (C) 2019 Western Digital Corporation or its affiliates.
 *
 * Authors:
 *     Anup Patel <anup.patel@wdc.com>
 */

#include <linux/errno.h>
#include <linux/err.h>
#include <linux/module.h>
#include <linux/kvm_host.h>
#include <asm/csr.h>
#include <asm/hwcap.h>
#include <asm/sbi.h>

long kvm_arch_dev_ioctl(struct file *filp,
			unsigned int ioctl, unsigned long arg)
{
	return -EINVAL;
}

int kvm_arch_hardware_enable(void)
{
	unsigned long hideleg, hedeleg;

	hedeleg = 0;
	hedeleg |= (1UL << EXC_INST_MISALIGNED);
	hedeleg |= (1UL << EXC_BREAKPOINT);
	hedeleg |= (1UL << EXC_SYSCALL);
	hedeleg |= (1UL << EXC_INST_PAGE_FAULT);
	hedeleg |= (1UL << EXC_LOAD_PAGE_FAULT);
	hedeleg |= (1UL << EXC_STORE_PAGE_FAULT);
	csr_write(CSR_HEDELEG, hedeleg);

	hideleg = 0;
	hideleg |= (1UL << IRQ_VS_SOFT);
	hideleg |= (1UL << IRQ_VS_TIMER);
	hideleg |= (1UL << IRQ_VS_EXT);
	csr_write(CSR_HIDELEG, hideleg);

	/* VS should access only the time counter directly. Everything else should trap */
	csr_write(CSR_HCOUNTEREN, 0x02);

	csr_write(CSR_HVIP, 0);

	kvm_riscv_aia_enable();

	return 0;
}

void kvm_arch_hardware_disable(void)
{
	kvm_riscv_aia_disable();

	/*
	 * After clearing the hideleg CSR, the host kernel will receive
	 * spurious interrupts if hvip CSR has pending interrupts and the
	 * corresponding enable bits in vsie CSR are asserted. To avoid it,
	 * hvip CSR and vsie CSR must be cleared before clearing hideleg CSR.
	 */
	csr_write(CSR_VSIE, 0);
	csr_write(CSR_HVIP, 0);
	csr_write(CSR_HEDELEG, 0);
	csr_write(CSR_HIDELEG, 0);
}

/**
 * @brief riscv 中kvm模块的入口函数
 **/
static int __init riscv_kvm_init(void)
{
	int rc;
	const char *str;

	/*1. 检查 RISC-V Hypervisor 扩展是否可用
	 */
	if (!riscv_isa_extension_available(NULL, h)) {
		kvm_info("hypervisor extension not available\n");
		return -ENODEV;
	}

	/*2. 检查 SBI 版本
	 */
	if (sbi_spec_is_0_1()) {
		kvm_info("require SBI v0.2 or higher\n");
		return -ENODEV;
	}

	/*3. 检查 SBI RFENCE 扩展是否可用
	 */
	if (!sbi_probe_extension(SBI_EXT_RFENCE)) {
		kvm_info("require SBI RFENCE extension\n");
		return -ENODEV;
	}

	/*4. 检测并设置 G-stage 页表模式
	 */
	kvm_riscv_gstage_mode_detect();//检测并配置二级页表模式

	kvm_riscv_gstage_vmid_detect();//检测并配置 VMID（虚拟机标识符）

	/*5. 初始化 AIA 以及中断虚拟化的一些全局参数;
	 *   AIA 用于管理更高级的中断功能
	 */
	rc = kvm_riscv_aia_init();
	if (rc && rc != -ENODEV)
		return rc;

	kvm_info("hypervisor extension available\n");

	/*6. 配置 G-stage 页表格式
	 */
	switch (kvm_riscv_gstage_mode()) {
	case HGATP_MODE_SV32X4:
		str = "Sv32x4";
		break;
	case HGATP_MODE_SV39X4:
		str = "Sv39x4";
		break;
	case HGATP_MODE_SV48X4:
		str = "Sv48x4";
		break;
	case HGATP_MODE_SV57X4:
		str = "Sv57x4";
		break;
	default:
		return -ENODEV;
	}
	kvm_info("using %s G-stage page table format\n", str);//打印系统中可用的 VMID 位数

	kvm_info("VMID %ld bits available\n", kvm_riscv_gstage_vmid_bits());

	/*7. 如果 AIA 可用，打印支持的外部中断数量
	 */
	if (kvm_riscv_aia_available())
		kvm_info("AIA available with %d guest external interrupts\n",
			 kvm_riscv_aia_nr_hgei);

	/*8. 调用 kvm_init 初始化 KVM 核心模块
	 */
	rc = kvm_init(sizeof(struct kvm_vcpu), 0, THIS_MODULE);
	if (rc) {
		kvm_riscv_aia_exit();
		return rc;
	}

	return 0;
}
module_init(riscv_kvm_init);

static void __exit riscv_kvm_exit(void)
{
	kvm_riscv_aia_exit();

	kvm_exit();
}
module_exit(riscv_kvm_exit);
