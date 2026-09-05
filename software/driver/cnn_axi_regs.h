#ifndef CNN_AXI_REGS_H
#define CNN_AXI_REGS_H

/*
 * cnn_axi_ctrl register offsets and masks.
 * Base address comes from Vivado Address Editor / xparameters.h — not defined here.
 */

#define CNN_CONTROL_OFFSET            0x00u
#define CNN_STATUS_OFFSET             0x04u
#define CNN_PREDICTED_CLASS_OFFSET    0x08u
#define CNN_MAX_LOGIT_OFFSET          0x0Cu

#define CNN_LOGIT0_OFFSET             0x10u
#define CNN_LOGIT1_OFFSET             0x14u
#define CNN_LOGIT2_OFFSET             0x18u
#define CNN_LOGIT3_OFFSET             0x1Cu
#define CNN_LOGIT4_OFFSET             0x20u

#define CNN_CYCLE_COUNT_LOW_OFFSET    0x24u
#define CNN_CYCLE_COUNT_HIGH_OFFSET   0x28u
#define CNN_RESERVED_2C_OFFSET        0x2Cu

/* Alias for software that only needs the low half name from the plan */
#define CNN_CYCLE_COUNT_OFFSET        CNN_CYCLE_COUNT_LOW_OFFSET

#define CNN_CONTROL_START_MASK        0x00000001u

#define CNN_STATUS_BUSY_MASK          0x00000001u
#define CNN_STATUS_DONE_MASK          0x00000002u

#endif /* CNN_AXI_REGS_H */
