#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef TOP_HEADER
#error "Define TOP_HEADER"
#endif
#ifndef TOP_CLASS
#error "Define TOP_CLASS"
#endif

#include TOP_HEADER

// Required by older Verilator $time / $display support.
double sc_time_stamp() { return 0; }

static void tick(TOP_CLASS* top) {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    TOP_CLASS* top = new TOP_CLASS;

    top->clk = 0;
    top->eval();

    int guard = 0;
    while (!Verilated::gotFinish()) {
        tick(top);
        if (++guard > 100000000) {
            std::fprintf(stderr, "TIMEOUT: simulation exceeded cycle guard\n");
            std::exit(1);
        }
    }

    delete top;
    return 0;
}
