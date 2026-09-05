#ifndef CNN_ACCELERATOR_H
#define CNN_ACCELERATOR_H

/*
 * Soft driver for cnn_axi_ctrl (AXI4-Lite) on Zynq PS.
 * Register contract matches docs/cnn_axi_register_map.md.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CNN_CONTROL_OFFSET           0x00u
#define CNN_STATUS_OFFSET            0x04u
#define CNN_PREDICTED_CLASS_OFFSET   0x08u
#define CNN_MAX_LOGIT_OFFSET         0x0Cu

#define CNN_LOGIT0_OFFSET            0x10u
#define CNN_LOGIT1_OFFSET            0x14u
#define CNN_LOGIT2_OFFSET            0x18u
#define CNN_LOGIT3_OFFSET            0x1Cu
#define CNN_LOGIT4_OFFSET            0x20u

#define CNN_CYCLE_COUNT_LOW_OFFSET   0x24u
#define CNN_CYCLE_COUNT_HIGH_OFFSET  0x28u

#define CNN_INPUT_WRITE_OFFSET       0x2Cu

#define CNN_CONTROL_START_MASK       0x1u

#define CNN_STATUS_BUSY_MASK         0x1u
#define CNN_STATUS_DONE_MASK         0x2u

#define CNN_INPUT_ADDR_MASK          0x00000FFFu
#define CNN_INPUT_DATA_SHIFT         12u
#define CNN_INPUT_LENGTH             3072u

void cnn_start(uintptr_t base);

uint32_t cnn_read_status(uintptr_t base);

int cnn_is_busy(uintptr_t base);

int cnn_is_done(uintptr_t base);

void cnn_write_input(uintptr_t base, uint16_t address, int8_t value);

/* Returns 0 on success, -1 if length != 3072 or input is NULL. Does not START. */
int cnn_load_input(uintptr_t base, const int8_t *input, size_t length);

uint32_t cnn_read_predicted_class(uintptr_t base);

int32_t cnn_read_max_logit(uintptr_t base);

/* class_index must be 0..4; returns 0 if out of range. */
int32_t cnn_read_logit(uintptr_t base, unsigned class_index);

uint64_t cnn_read_cycle_count(uintptr_t base);

#ifdef __cplusplus
}
#endif

#endif /* CNN_ACCELERATOR_H */
