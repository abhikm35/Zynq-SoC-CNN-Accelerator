#include "cnn_accelerator.h"

#include "xil_io.h"

void cnn_start(uintptr_t base)
{
    Xil_Out32(base + CNN_CONTROL_OFFSET, CNN_CONTROL_START_MASK);
}

uint32_t cnn_read_status(uintptr_t base)
{
    return Xil_In32(base + CNN_STATUS_OFFSET);
}

int cnn_is_busy(uintptr_t base)
{
    return (cnn_read_status(base) & CNN_STATUS_BUSY_MASK) != 0u ? 1 : 0;
}

int cnn_is_done(uintptr_t base)
{
    return (cnn_read_status(base) & CNN_STATUS_DONE_MASK) != 0u ? 1 : 0;
}

void cnn_write_input(uintptr_t base, uint16_t address, int8_t value)
{
    uint32_t packed =
        ((uint32_t)address & CNN_INPUT_ADDR_MASK) |
        (((uint32_t)(uint8_t)value) << CNN_INPUT_DATA_SHIFT);

    Xil_Out32(base + CNN_INPUT_WRITE_OFFSET, packed);
}

int cnn_load_input(uintptr_t base, const int8_t *input, size_t length)
{
    size_t i;

    if (input == NULL || length != CNN_INPUT_LENGTH) {
        return -1;
    }

    for (i = 0; i < length; ++i) {
        cnn_write_input(base, (uint16_t)i, input[i]);
    }

    return 0;
}

uint32_t cnn_read_predicted_class(uintptr_t base)
{
    return Xil_In32(base + CNN_PREDICTED_CLASS_OFFSET) & 0x7u;
}

int32_t cnn_read_max_logit(uintptr_t base)
{
    return (int32_t)Xil_In32(base + CNN_MAX_LOGIT_OFFSET);
}

int32_t cnn_read_logit(uintptr_t base, unsigned class_index)
{
    static const uint32_t offsets[5] = {
        CNN_LOGIT0_OFFSET,
        CNN_LOGIT1_OFFSET,
        CNN_LOGIT2_OFFSET,
        CNN_LOGIT3_OFFSET,
        CNN_LOGIT4_OFFSET
    };

    if (class_index > 4u) {
        return 0;
    }

    return (int32_t)Xil_In32(base + offsets[class_index]);
}

uint64_t cnn_read_cycle_count(uintptr_t base)
{
    uint32_t low = Xil_In32(base + CNN_CYCLE_COUNT_LOW_OFFSET);
    uint32_t high = Xil_In32(base + CNN_CYCLE_COUNT_HIGH_OFFSET);

    return ((uint64_t)high << 32) | (uint64_t)low;
}
