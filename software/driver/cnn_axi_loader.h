#ifndef CNN_AXI_LOADER_H
#define CNN_AXI_LOADER_H

/*
 * Soft helpers for cnn_axi_ctrl Activation RAM A input loading.
 *
 * Before including this header in a Vitis/standalone app:
 *
 *   #include "xil_io.h"
 *   #define CNN_AXI_OUT32(addr, data) Xil_Out32((UINTPTR)(addr), (u32)(data))
 *   #include "cnn_axi_loader.h"
 *
 * Host/unit-test builds may define CNN_AXI_OUT32 to any MMIO write.
 */

#include <stddef.h>
#include <stdint.h>

#include "cnn_axi_regs.h"

#ifndef CNN_AXI_OUT32
#error "Define CNN_AXI_OUT32(addr, data) before including cnn_axi_loader.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline void cnn_write_input_byte(uintptr_t base, uint16_t address, int8_t value)
{
    CNN_AXI_OUT32(base + CNN_INPUT_ADDRESS_OFFSET, (uint32_t)(address & 0xFFFu));
    CNN_AXI_OUT32(base + CNN_INPUT_DATA_OFFSET, (uint32_t)(uint8_t)value);
    CNN_AXI_OUT32(base + CNN_INPUT_COMMAND_OFFSET, CNN_INPUT_WRITE_MASK);
}

/* Load CHW INT8 tensor (length must be 3072). Returns 0 on success, -1 on bad length. */
static inline int cnn_load_input(uintptr_t base, const int8_t *image, size_t length)
{
    size_t i;

    if (image == NULL || length != CNN_INPUT_LENGTH)
        return -1;

    for (i = 0; i < length; ++i)
        cnn_write_input_byte(base, (uint16_t)i, image[i]);

    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* CNN_AXI_LOADER_H */
