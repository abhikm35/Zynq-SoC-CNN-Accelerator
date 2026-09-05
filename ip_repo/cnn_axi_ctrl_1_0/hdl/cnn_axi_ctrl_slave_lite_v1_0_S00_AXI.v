
`timescale 1 ns / 1 ps

	module cnn_axi_ctrl_slave_lite_v1_0_S00_AXI #
	(
		// Users to add parameters here
		parameter integer C_CNN_LOGIT_WIDTH = 32,
		// User parameters ends
		// Do not modify the parameters beyond this line

		// Width of S_AXI data bus
		parameter integer C_S_AXI_DATA_WIDTH	= 32,
		// Width of S_AXI address bus
		parameter integer C_S_AXI_ADDR_WIDTH	= 6
	)
	(
		// Users to add ports here
		// CNN accelerator control / status (same PL clock domain assumed)
		output wire                              cnn_start,
		input  wire                              cnn_busy,
		input  wire                              cnn_done,
		input  wire [2:0]                        cnn_predicted_class,
		input  wire signed [C_CNN_LOGIT_WIDTH-1:0] cnn_maximum_logit,
		input  wire signed [C_CNN_LOGIT_WIDTH-1:0] cnn_logit_0,
		input  wire signed [C_CNN_LOGIT_WIDTH-1:0] cnn_logit_1,
		input  wire signed [C_CNN_LOGIT_WIDTH-1:0] cnn_logit_2,
		input  wire signed [C_CNN_LOGIT_WIDTH-1:0] cnn_logit_3,
		input  wire signed [C_CNN_LOGIT_WIDTH-1:0] cnn_logit_4,
		input  wire [63:0]                       cnn_cycle_count,
		// Activation RAM A input-tensor load (AXI4-Lite loader; not BRAM/DMA)
		output wire                              cnn_input_write_enable,
		output wire [11:0]                       cnn_input_write_address,
		output wire [7:0]                        cnn_input_write_data,
		// User ports ends
		// Do not modify the ports beyond this line

		// Global Clock Signal
		input wire  S_AXI_ACLK,
		// Global Reset Signal. This Signal is Active LOW
		input wire  S_AXI_ARESETN,
		// Write address (issued by master, acceped by Slave)
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
		// Write channel Protection type. This signal indicates the
    		// privilege and security level of the transaction, and whether
    		// the transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_AWPROT,
		// Write address valid. This signal indicates that the master signaling
    		// valid write address and control information.
		input wire  S_AXI_AWVALID,
		// Write address ready. This signal indicates that the slave is ready
    		// to accept an address and associated control signals.
		output wire  S_AXI_AWREADY,
		// Write data (issued by master, acceped by Slave) 
		input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
		// Write strobes. This signal indicates which byte lanes hold
    		// valid data. There is one write strobe bit for each eight
    		// bits of the write data bus.    
		input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
		// Write valid. This signal indicates that valid write
    		// data and strobes are available.
		input wire  S_AXI_WVALID,
		// Write ready. This signal indicates that the slave
    		// can accept the write data.
		output wire  S_AXI_WREADY,
		// Write response. This signal indicates the status
    		// of the write transaction.
		output wire [1 : 0] S_AXI_BRESP,
		// Write response valid. This signal indicates that the channel
    		// is signaling a valid write response.
		output wire  S_AXI_BVALID,
		// Response ready. This signal indicates that the master
    		// can accept a write response.
		input wire  S_AXI_BREADY,
		// Read address (issued by master, acceped by Slave)
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
		// Protection type. This signal indicates the privilege
    		// and security level of the transaction, and whether the
    		// transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_ARPROT,
		// Read address valid. This signal indicates that the channel
    		// is signaling valid read address and control information.
		input wire  S_AXI_ARVALID,
		// Read address ready. This signal indicates that the slave is
    		// ready to accept an address and associated control signals.
		output wire  S_AXI_ARREADY,
		// Read data (issued by slave)
		output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
		// Read response. This signal indicates the status of the
    		// read transfer.
		output wire [1 : 0] S_AXI_RRESP,
		// Read valid. This signal indicates that the channel is
    		// signaling the required read data.
		output wire  S_AXI_RVALID,
		// Read ready. This signal indicates that the master can
    		// accept the read data and response information.
		input wire  S_AXI_RREADY
	);

	// AXI4LITE signals
	reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr;
	reg  	axi_awready;
	reg  	axi_wready;
	reg [1 : 0] 	axi_bresp;
	reg  	axi_bvalid;
	reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr;
	reg  	axi_arready;
	reg [1 : 0] 	axi_rresp;
	reg  	axi_rvalid;

	// local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
	// ADDR_LSB is used for addressing 32/64 bit registers/memories
	// ADDR_LSB = 2 for 32 bits (n downto 2)
	// ADDR_LSB = 3 for 64 bits (n downto 3)
	localparam integer ADDR_LSB = (C_S_AXI_DATA_WIDTH/32) + 1;
	localparam integer OPT_MEM_ADDR_BITS = 3;

	// I/O Connections assignments

	assign S_AXI_AWREADY	= axi_awready;
	assign S_AXI_WREADY	= axi_wready;
	assign S_AXI_BRESP	= axi_bresp;
	assign S_AXI_BVALID	= axi_bvalid;
	assign S_AXI_ARREADY	= axi_arready;
	assign S_AXI_RRESP	= axi_rresp;
	assign S_AXI_RVALID	= axi_rvalid;
	 //state machine varibles 
	 reg [1:0] state_write;
	 reg [1:0] state_read;
	 //State machine local parameters
	 localparam Idle = 2'b00,Raddr = 2'b10,Rdata = 2'b11 ,Waddr = 2'b10,Wdata = 2'b11;
	// Implement Write state machine
	// Outstanding write transactions are not supported by the slave i.e., master should assert bready to receive response on or before it starts sending the new transaction
	always @(posedge S_AXI_ACLK)                                 
	  begin                                 
	     if (S_AXI_ARESETN == 1'b0)                                 
	       begin                                 
	         axi_awready <= 0;                                 
	         axi_wready <= 0;                                 
	         axi_bvalid <= 0;                                 
	         axi_bresp <= 0;                                 
	         axi_awaddr <= 0;                                 
	         state_write <= Idle;                                 
	       end                                 
	     else                                  
	       begin                                 
	         case(state_write)                                 
	           Idle:                                      
	             begin                                 
	               if(S_AXI_ARESETN == 1'b1)                                  
	                 begin                                 
	                   axi_awready <= 1'b1;                                 
	                   axi_wready <= 1'b1;                                 
	                   state_write <= Waddr;                                 
	                 end                                 
	               else state_write <= state_write;                                 
	             end                                 
	           Waddr:        //At this state, slave is ready to receive address along with corresponding control signals and first data packet. Response valid is also handled at this state                                 
	             begin                                 
	               if (S_AXI_AWVALID && S_AXI_AWREADY)                                 
	                  begin                                 
	                    axi_awaddr <= S_AXI_AWADDR;                                 
	                    if(S_AXI_WVALID)                                  
	                      begin                                   
	                        axi_awready <= 1'b1;                                 
	                        state_write <= Waddr;                                 
	                        axi_bvalid <= 1'b1;                                 
	                      end                                 
	                    else                                  
	                      begin                                 
	                        axi_awready <= 1'b0;                                 
	                        state_write <= Wdata;                                 
	                        if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;                                 
	                      end                                 
	                  end                                 
	               else                                  
	                  begin                                 
	                    state_write <= state_write;                                 
	                    if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;                                 
	                   end                                 
	             end                                 
	          Wdata:        //At this state, slave is ready to receive the data packets until the number of transfers is equal to burst length                                 
	             begin                                 
	               if (S_AXI_WVALID)                                 
	                 begin                                 
	                   state_write <= Waddr;                                 
	                   axi_bvalid <= 1'b1;                                 
	                   axi_awready <= 1'b1;                                 
	                 end                                 
	                else                                  
	                 begin                                 
	                   state_write <= state_write;                                 
	                   if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;                                 
	                 end                                              
	             end                                 
	          endcase                                 
	        end                                 
	      end                                 

	//----------------------------------------------------------------------
	// CNN control / status user logic
	//----------------------------------------------------------------------
	// AXI write commit: same moments the generated template accepts WDATA
	// into the register map (Waddr with AW+W, or Wdata with W).
	wire axi_write_commit =
	    S_AXI_WVALID && S_AXI_WREADY &&
	    ( (state_write == Waddr && S_AXI_AWVALID && S_AXI_AWREADY) ||
	      (state_write == Wdata) );

	wire [C_S_AXI_ADDR_WIDTH-1:0] wr_addr_bus =
	    S_AXI_AWVALID ? S_AXI_AWADDR : axi_awaddr;

	wire [OPT_MEM_ADDR_BITS:0] wr_reg_sel =
	    wr_addr_bus[ADDR_LSB+OPT_MEM_ADDR_BITS : ADDR_LSB];

	// CONTROL @ 0x00 / sel 0: START is a one-cycle pulse, not a sticky bit.
	// Honor WSTRB[0]. Ignore START while cnn_busy (do not restart).
	wire control_start_cmd =
	    axi_write_commit &&
	    (wr_reg_sel == 4'h0) &&
	    S_AXI_WSTRB[0] &&
	    S_AXI_WDATA[0];

	wire start_accepted = control_start_cmd && (cnn_busy == 1'b0);

	reg cnn_start_r;
	reg done_sticky;

	assign cnn_start = cnn_start_r;

	//----------------------------------------------------------------------
	// INPUT_WRITE @ 0x2C / sel 4'hB — packed one-shot Activation RAM A write
	//   [11:0]  = address (valid 0..3071)
	//   [19:12] = INT8 data bits
	//   [31:20] = reserved
	// Requires WSTRB[2:0]=3'b111. Ignored if cnn_busy or address >= 3072.
	//----------------------------------------------------------------------
	localparam [11:0] CNN_INPUT_LEN = 12'd3072;

	reg [11:0] input_address_reg;
	reg [7:0]  input_data_reg;
	reg        cnn_input_write_enable_r;

	wire input_write_cmd =
	    axi_write_commit &&
	    (wr_reg_sel == 4'hB) &&
	    S_AXI_WSTRB[0] && S_AXI_WSTRB[1] && S_AXI_WSTRB[2];

	wire input_addr_in_range = (S_AXI_WDATA[11:0] < CNN_INPUT_LEN);

	wire input_write_accepted =
	    input_write_cmd &&
	    (cnn_busy == 1'b0) &&
	    input_addr_in_range;

	assign cnn_input_write_enable  = cnn_input_write_enable_r;
	assign cnn_input_write_address = input_address_reg;
	assign cnn_input_write_data    = input_data_reg;

	always @(posedge S_AXI_ACLK)
	begin
	  if (S_AXI_ARESETN == 1'b0) begin
	    cnn_start_r <= 1'b0;
	    done_sticky <= 1'b0;
	    input_address_reg <= 12'd0;
	    input_data_reg <= 8'd0;
	    cnn_input_write_enable_r <= 1'b0;
	  end else begin
	    // Default: pulses low every cycle unless accepting a command
	    cnn_start_r <= 1'b0;
	    cnn_input_write_enable_r <= 1'b0;

	    if (start_accepted) begin
	      cnn_start_r <= 1'b1;
	      done_sticky <= 1'b0;
	    end else if (cnn_done) begin
	      done_sticky <= 1'b1;
	    end

	    // Capture packed addr/data and assert WE for exactly one cycle
	    if (input_write_accepted) begin
	      input_address_reg <= S_AXI_WDATA[11:0];
	      input_data_reg <= S_AXI_WDATA[19:12];
	      cnn_input_write_enable_r <= 1'b1;
	    end
	  end
	end

	// Sign-extend CNN logits to 32-bit AXI RDATA (LOGIT_WIDTH may be <= 32)
	wire signed [31:0] max_logit_sx =
	    {{(32-C_CNN_LOGIT_WIDTH){cnn_maximum_logit[C_CNN_LOGIT_WIDTH-1]}}, cnn_maximum_logit};
	wire signed [31:0] logit0_sx =
	    {{(32-C_CNN_LOGIT_WIDTH){cnn_logit_0[C_CNN_LOGIT_WIDTH-1]}}, cnn_logit_0};
	wire signed [31:0] logit1_sx =
	    {{(32-C_CNN_LOGIT_WIDTH){cnn_logit_1[C_CNN_LOGIT_WIDTH-1]}}, cnn_logit_1};
	wire signed [31:0] logit2_sx =
	    {{(32-C_CNN_LOGIT_WIDTH){cnn_logit_2[C_CNN_LOGIT_WIDTH-1]}}, cnn_logit_2};
	wire signed [31:0] logit3_sx =
	    {{(32-C_CNN_LOGIT_WIDTH){cnn_logit_3[C_CNN_LOGIT_WIDTH-1]}}, cnn_logit_3};
	wire signed [31:0] logit4_sx =
	    {{(32-C_CNN_LOGIT_WIDTH){cnn_logit_4[C_CNN_LOGIT_WIDTH-1]}}, cnn_logit_4};

	wire [31:0] predicted_class_zx = {29'b0, cnn_predicted_class};
	wire [31:0] status_word = {30'b0, done_sticky, cnn_busy};

	// Register map (word select = offset[5:2]):
	// 0x00 CONTROL           -> 0 (START is write-one command only)
	// 0x04 STATUS            -> {done_sticky, busy}
	// 0x08 PREDICTED_CLASS
	// 0x0C MAX_LOGIT
	// 0x10..0x20 LOGIT_0..4
	// 0x24 CYCLE_COUNT_LOW
	// 0x28 CYCLE_COUNT_HIGH
	// 0x2C INPUT_WRITE       -> 0 (write-only packed command)
	wire [OPT_MEM_ADDR_BITS:0] rd_reg_sel =
	    axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS : ADDR_LSB];

	reg [C_S_AXI_DATA_WIDTH-1:0] reg_data_out;
	always @(*)
	begin
	  case (rd_reg_sel)
	    4'h0: reg_data_out = 32'h0000_0000;
	    4'h1: reg_data_out = status_word;
	    4'h2: reg_data_out = predicted_class_zx;
	    4'h3: reg_data_out = max_logit_sx;
	    4'h4: reg_data_out = logit0_sx;
	    4'h5: reg_data_out = logit1_sx;
	    4'h6: reg_data_out = logit2_sx;
	    4'h7: reg_data_out = logit3_sx;
	    4'h8: reg_data_out = logit4_sx;
	    4'h9: reg_data_out = cnn_cycle_count[31:0];
	    4'hA: reg_data_out = cnn_cycle_count[63:32];
	    4'hB: reg_data_out = 32'h0000_0000; // INPUT_WRITE (WO)
	    default: reg_data_out = 32'h0000_0000;
	  endcase
	end

	assign S_AXI_RDATA = reg_data_out;

	// Implement read state machine
	  always @(posedge S_AXI_ACLK)                                       
	    begin                                       
	      if (S_AXI_ARESETN == 1'b0)                                       
	        begin                                       
	         //asserting initial values to all 0's during reset                                       
	         axi_arready <= 1'b0;                                       
	         axi_rvalid <= 1'b0;                                       
	         axi_rresp <= 1'b0;                                       
	         state_read <= Idle;                                       
	        end                                       
	      else                                       
	        begin                                       
	          case(state_read)                                       
	            Idle:     //Initial state inidicating reset is done and ready to receive read/write transactions                                       
	              begin                                                
	                if (S_AXI_ARESETN == 1'b1)                                        
	                  begin                                       
	                    state_read <= Raddr;                                       
	                    axi_arready <= 1'b1;                                       
	                  end                                       
	                else state_read <= state_read;                                       
	              end                                       
	            Raddr:        //At this state, slave is ready to receive address along with corresponding control signals                                       
	              begin                                       
	                if (S_AXI_ARVALID && S_AXI_ARREADY)                                       
	                  begin                                       
	                    state_read <= Rdata;                                       
	                    axi_araddr <= S_AXI_ARADDR;                                       
	                    axi_rvalid <= 1'b1;                                       
	                    axi_arready <= 1'b0;                                       
	                  end                                       
	                else state_read <= state_read;                                       
	              end                                       
	            Rdata:        //At this state, slave is ready to send the data packets until the number of transfers is equal to burst length                                       
	              begin                                           
	                if (S_AXI_RVALID && S_AXI_RREADY)                                       
	                  begin                                       
	                    axi_rvalid <= 1'b0;                                       
	                    axi_arready <= 1'b1;                                       
	                    state_read <= Raddr;                                       
	                  end                                       
	                else state_read <= state_read;                                       
	              end                                       
	           endcase                                       
	          end                                       
	        end                                         

	endmodule
