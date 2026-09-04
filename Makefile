.PHONY: test-rtl-single-output test-rtl-memory-output test-rtl-conv1-channel \
	test-rtl-conv1-full test-rtl-maxpool1 test-rtl-conv2 test-rtl-maxpool2 \
	test-rtl-gap test-rtl-fc test-rtl-argmax test-rtl-end-to-end \
	test-rtl-pingpong test-rtl-shared-conv test-rtl-shared-pool \
	export-rtl-vectors export-conv1-memory export-conv1-channel \
	export-conv1-full export-pool1 export-conv2 export-pool2 export-gap \
	export-fc export-argmax export-end-to-end

export-rtl-vectors:
	PYTHONPATH=. python -m tools.export_rtl_vectors

export-conv1-memory:
	PYTHONPATH=. python -m tools.export_conv1_memory_vectors

export-conv1-channel:
	PYTHONPATH=. python -m tools.export_conv1_channel_vectors

export-conv1-full:
	PYTHONPATH=. python -m tools.export_conv1_full_vectors

export-pool1:
	PYTHONPATH=. python -m tools.export_pool1_vectors

export-conv2:
	PYTHONPATH=. python -m tools.export_conv2_vectors

export-pool2:
	PYTHONPATH=. python -m tools.export_pool2_vectors

export-gap:
	PYTHONPATH=. python -m tools.export_gap_vectors

export-fc:
	PYTHONPATH=. python -m tools.export_fc_vectors

export-argmax:
	PYTHONPATH=. python -m tools.export_argmax_vectors

export-end-to-end:
	PYTHONPATH=. python -m tools.export_end_to_end_vectors

test-rtl-single-output:
	PYTHONPATH=. python -m tools.run_rtl_tests --only arith

test-rtl-memory-output:
	PYTHONPATH=. python -m tools.run_rtl_tests --only memory

test-rtl-conv1-channel:
	PYTHONPATH=. python -m tools.run_rtl_tests --only channel

# Full suite including Conv1 .. Argmax and end-to-end
test-rtl-conv1-full:
	PYTHONPATH=. python -m tools.run_rtl_tests

test-rtl-maxpool1:
	PYTHONPATH=. python -m tools.run_rtl_tests --only pool

test-rtl-conv2:
	PYTHONPATH=. python -m tools.run_rtl_tests --only conv2

test-rtl-maxpool2:
	PYTHONPATH=. python -m tools.run_rtl_tests --only pool2

test-rtl-gap:
	PYTHONPATH=. python -m tools.run_rtl_tests --only gap

test-rtl-fc:
	PYTHONPATH=. python -m tools.run_rtl_tests --only fc

test-rtl-argmax:
	PYTHONPATH=. python -m tools.run_rtl_tests --only argmax

test-rtl-end-to-end:
	PYTHONPATH=. python -m tools.run_rtl_tests --only e2e

test-rtl-pingpong:
	PYTHONPATH=. python -m tools.run_rtl_tests --only pingpong

test-rtl-shared-conv:
	PYTHONPATH=. python -m tools.run_rtl_tests --only shared_conv

test-rtl-shared-pool:
	PYTHONPATH=. python -m tools.run_rtl_tests --only shared_pool
