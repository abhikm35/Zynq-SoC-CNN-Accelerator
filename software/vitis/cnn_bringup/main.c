/*
 * cnn_bringup — first Arty Z7-10 FPGA inference vs INT8 Python / RTL golden.
 *
 * Edit this file in Git (Mac/Cursor). On Windows, Vitis must reference this
 * path directly (do not copy into vitis_workspace).
 */

#include "xil_printf.h"
#include "xparameters.h"

#include "cnn_accelerator.h"
#include "known_test_image.h"

/*
 * Current Vivado Address Editor assignment:
 *   cnn_axi_ctrl_0 / S00_AXI = 0x43C00000
 *
 * Windows Vitis should preferably replace this with the generated
 * xparameters.h macro once its exact name is confirmed (often similar to
 * XPAR_CNN_AXI_CTRL_0_S00_AXI_BASEADDR or XPAR_CNN_AXI_CTRL_0_BASEADDR).
 * Until that name is verified on the Windows platform, use this default.
 */
#ifndef CNN_BASE_ADDRESS
#define CNN_BASE_ADDRESS 0x43C00000u
#endif

/* Generous bare-metal poll budget (~seconds of busy-wait at ARM speed). */
#ifndef CNN_DONE_TIMEOUT
#define CNN_DONE_TIMEOUT 200000000u
#endif

static void print_u64(uint64_t value)
{
    char buf[21];
    int i = 0;
    int j;

    if (value == 0ull) {
        xil_printf("0");
        return;
    }

    while (value > 0ull && i < 20) {
        buf[i++] = (char)('0' + (int)(value % 10ull));
        value /= 10ull;
    }

    for (j = i - 1; j >= 0; --j) {
        xil_printf("%c", buf[j]);
    }
}

int main(void)
{
    const uintptr_t base = (uintptr_t)CNN_BASE_ADDRESS;
    uint32_t status;
    uint32_t timeout;
    uint32_t predicted;
    int32_t max_logit;
    int32_t logits[5];
    uint64_t cycles;
    unsigned i;
    int load_rc;
    int logits_ok;
    int class_ok;
    int max_ok;
    int cycles_ok;
    int overall_ok;

    xil_printf("\r\n");
    xil_printf("CNN FPGA Bring-Up\r\n");
    xil_printf("-----------------\r\n");
    xil_printf("CNN base address: 0x%08x\r\n", (unsigned)CNN_BASE_ADDRESS);

    status = cnn_read_status(base);
    xil_printf("Initial STATUS: 0x%08x\r\n", (unsigned)status);

    if (cnn_is_busy(base)) {
        xil_printf("ERROR: accelerator busy before load (STATUS=0x%08x)\r\n",
                   (unsigned)status);
        xil_printf("OVERALL: FAIL\r\n");
        return 1;
    }

    xil_printf("Loading %u INT8 activations...\r\n",
               (unsigned)KNOWN_TEST_IMAGE_LENGTH);
    load_rc = cnn_load_input(base, known_test_image, KNOWN_TEST_IMAGE_LENGTH);
    if (load_rc != 0) {
        xil_printf("ERROR: cnn_load_input failed (%d)\r\n", load_rc);
        xil_printf("OVERALL: FAIL\r\n");
        return 1;
    }
    xil_printf("Input loaded.\r\n");

    xil_printf("Starting CNN...\r\n");
    cnn_start(base);

    timeout = CNN_DONE_TIMEOUT;
    while (!cnn_is_done(base) && timeout != 0u) {
        --timeout;
    }

    if (timeout == 0u) {
        status = cnn_read_status(base);
        xil_printf("ERROR: DONE timeout (STATUS=0x%08x)\r\n", (unsigned)status);
        xil_printf("OVERALL: FAIL\r\n");
        return 1;
    }

    xil_printf("CNN complete.\r\n");

    predicted = cnn_read_predicted_class(base);
    max_logit = cnn_read_max_logit(base);
    for (i = 0; i < 5u; ++i) {
        logits[i] = cnn_read_logit(base, i);
    }
    cycles = cnn_read_cycle_count(base);

    xil_printf("\r\nLogits:\r\n");
    for (i = 0; i < 5u; ++i) {
        xil_printf("%u: %d\r\n", i, (int)logits[i]);
    }
    xil_printf("\r\nMaximum logit: %d\r\n", (int)max_logit);
    xil_printf("Predicted class: %u\r\n", (unsigned)predicted);
    xil_printf("Cycle count: ");
    print_u64(cycles);
    xil_printf("\r\n");

    logits_ok = 1;
    if (logits[0] != KNOWN_EXPECTED_LOGIT_0 ||
        logits[1] != KNOWN_EXPECTED_LOGIT_1 ||
        logits[2] != KNOWN_EXPECTED_LOGIT_2 ||
        logits[3] != KNOWN_EXPECTED_LOGIT_3 ||
        logits[4] != KNOWN_EXPECTED_LOGIT_4) {
        logits_ok = 0;
    }

    class_ok = (predicted == KNOWN_EXPECTED_PREDICTED_CLASS) ? 1 : 0;
    max_ok = (max_logit == KNOWN_EXPECTED_MAX_LOGIT) ? 1 : 0;

#if KNOWN_HAS_EXPECTED_CYCLE_COUNT
    cycles_ok = (cycles == KNOWN_EXPECTED_CYCLE_COUNT) ? 1 : 0;
#else
    cycles_ok = 1; /* no golden cycle count to compare */
#endif

    xil_printf("\r\nGolden comparison:\r\n");
    xil_printf("logits: %s\r\n", logits_ok ? "PASS" : "FAIL");
    xil_printf("max_logit: %s\r\n", max_ok ? "PASS" : "FAIL");
    xil_printf("class: %s\r\n", class_ok ? "PASS" : "FAIL");
#if KNOWN_HAS_EXPECTED_CYCLE_COUNT
    xil_printf("cycle_count: %s\r\n", cycles_ok ? "PASS" : "FAIL");
#else
    xil_printf("cycle_count: SKIP (no golden)\r\n");
#endif

    overall_ok = logits_ok && class_ok && max_ok && cycles_ok;
    xil_printf("\r\nOVERALL: %s\r\n", overall_ok ? "PASS" : "FAIL");

    return overall_ok ? 0 : 1;
}
