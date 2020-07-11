`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Engineer:        Sebastian Thiem
// 
// Create Date:     15 February 2019 
// File Name:       datapath.v 
// Project Name:    MIPS 32-bit 5-stage Pipelined Processor
//
// Module(s):       datapath
// Description(s):  datapath - A top module for the MIPS processor that 
//                          lays out the datapath 
//
//////////////////////////////////////////////////////////////////////////////////
module Datapath(CLK, RST,
                        OUTVAL, HI, LO, REGADDR, REGWRITE, A, B);
    
    /*
    Controller[24] = branch
    Controller[23:20] = branchLogic
    Controller[19] = regWrite
    Controller[18:17] = memToReg
    Controller[16] = memWrite
    Controller[15] = memRead
    Controller[14] = ALUSrcA
    Controller[13] = ALUSrcB
    Controller[12] = ALUOutMux
    Controller[11] = signMux
    Controller[10] = hiLoMux
    Controller[9:8] = DataSelWrite 
    Controller[7:3] = ALUControl
    Controller[2] = regDest
    Controller[1:0] = movSel
    */
    
    input CLK;
    input RST;
    
    output [31:0] OUTVAL;
    output [31:0] HI;
    output [31:0] LO;
    output [4:0] REGADDR;
    output REGWRITE;
    output[31:0] A;
    output[31:0] B;
    
    //Instruction Fetch****************************
    wire[31:0] PCAddResult;    // Ouput of PCAdder
    wire[31:0] Instruction;    // Output of instructionMemory
    wire[31:0] PCResult;       // Output of ProgramCounter
    wire[31:0] PCSrcOut;       // Output of the mux before ProgramCounter

    //IF/ID RegOut************************************
    wire[31:0] PCAddResult_ID;
    wire[31:0] Instruction_ID;
    
    //Instruction Decode***************************
    wire[24:0] Controller; // Output of the control module
    
    wire[24:0] ControllerMuxOut;   // ouput of mux hat choses between stall(1), and Controller bypass(0)
    
    wire ctrlMuxStall;  // Stalls the controller output NOP
    wire stall;     // Pauses the IF/ID reg and PC
    
    wire[31:0] readData1;  // Output of ReadData1
    wire[31:0] readData2;  // Output of ReadData2
    
    wire[31:0] readData1_jal;   // data path that can carry PCAddResult in linked jumps
    
    wire PCSrc;                 // is high when branching
    wire[31:0] branchAddress;   // calculated branching address
    wire jal;                   // is high when linking after a jump
    
    wire[31:0] signExtend;     // Output of the 16->32 sign extension
    wire[31:0] upperImmediate; // immidiate value in the MSB of a 32bit word
    
    wire[4:0] rt_jal;      // output of that mux that chooses between rt and $ra
    
    wire rsMuxCtrl_ID;     // control for the ID stage rs fwding mux MEM(1), ReadData1(0)
    wire rtMuxCtrl_ID;     // control for the ID stage rt fwding mux MEM(1), ReadData2(0)
    wire[31:0] rsFwdMuxOut_ID;    // output of the mux that choses between MEM(1), ReadData1(0)
    wire[31:0] rtFwdMuxOut_ID;    // output of the mux that choses between MEM(1), ReadData2(0)
    
    //ID/EX Reg Out************************************
    wire regWrite_EX;
    wire[1:0] memToReg_EX;
    wire[1:0] movCtrl_EX;
    wire[5:0] funct_EX;
    wire memWrite_EX; 
    wire memRead_EX;
    wire ALUSrcA_EX;
    wire ALUSrcB_EX;
    wire ALUOutMux_EX;
    wire signMux_EX;
    wire hiLoMux_EX;
    wire[1:0]DataSelWrite_EX;
    wire[4:0] ALUControl_EX; 
    wire regDst_EX; 
   
    wire[31:0] readData1_EX;
    wire[31:0] readData2_EX;
    wire[31:0] immediate_EX;
    wire[31:0] upperImmediate_EX;
    wire[4:0] rs_EX;
    wire[4:0] rt_EX;
    wire[4:0] rd_EX;
    
    //Execution************************************
    wire[31:0] ALUSrcAMuxOut;  // Mux output that chooses betwoon the rs(0) and rt(1) fields
    wire[31:0] ALUSrcBMuxOut;  // Mux output that chose between the rt(0) or the immediate(1) field
    wire[31:0] ABSAOut;        // Absolute value of A val for the ALU
    wire[31:0] ABSBOut;        // Absolute value of B val for the ALU
    wire[31:0] signMuxA;       // Output of the mux that choses between the A val, and the abs(A) val
    wire[31:0] signMuxB;       // Output of the mux that choses between the B val, and the abs(B) val
    
    wire[31:0] ALUResult;       // Result of ALU operation
    wire[63:0] thicc;          // Result for the ALU sent to te HiLo reg 
        
    wire[63:0] HiLoOut;         // Value stored in the HiLo reg, set into the ALU
    wire[31:0] hiloMuxOut;      // Output of the mux that choses between the lo(0) and hi(1) reg value
    wire[31:0] ALUOutMuxOut;    // Output of the mux that choses between the reg(0) and ALUResult(1)
    
    wire[31:0] regDataSEH;    // readData Half Extended
    wire[31:0] regDataSEB;    // readData Byte extended  
    
    wire[31:0] dataWriteMemMux;   // Output of a mux that choses between rt(0), half rt(1), byte rt(2), immi(3)
    
    wire[4:0] regDstOut;        // Outut of the mux that choses between rt(0) or rd(1)
    
    wire isMov;    // is 1 when its a mov and condition is met, else 0 
    wire movMux;   // Output of the mux that setreg write to true on mov operations
    
    wire[1:0] rsMuxCtrl_EX;    // Controls the EXE stage rs fwding mux 0(3), WB(2), MEM(1), ReadData1(0)
    wire[1:0] rtMuxCtrl_EX;    // Controls the EXE stage rt fwding mux 0(3), WB(2), MEM(1), ReadData2(0)
    wire[31:0] rsFwdMuxOut_EX;    // output of the mux that choses between 0(3), WB(2), MEM(1), ReadData1(0)
    wire[31:0] rtFwdMuxOut_EX;    // output of the mux that choses between 0(3), WB(2), MEM(1), ReadData2(0)
    
    //EX/MEM reg out***********************************
    wire regWrite_MEM;
    wire[1:0] memToReg_MEM;
    wire isMov_MEM;
    wire memWrite_MEM;
    wire memRead_MEM;
    wire[1:0] DataSelWrite_MEM;
    
    wire[31:0] branchAddress_EX; 
    wire PCSrc_EX;
    
    wire[31:0] EXEResult_MEM;
    wire[31:0] dataWriteMem_MEM;
    wire[31:0] upperImmediate_MEM;
    wire[4:0] regDstMuxOut_MEM;
    
    //Memory Access********************************
    wire[31:0] readData;  // Output of readData from Memory
    
    wire[31:0] memDataSEH;    // readData Half Extended
    wire[31:0] memDataSEB;    // readData Byte extended    
    
    wire[31:0] readDataMux;   // Output of a mux that choses between MEM(0), half MEM(1), byte MEM(2), 0(3)
    
    //MEM/WB reg out***********************************
    wire regWrite_WB;
    wire[1:0] memToReg_WB;
    wire isMov_WB;
    
    wire[31:0] branchAddress_MEM; 
    wire PCSrc_MEM;
    
    wire[31:0] EXEResult_WB;
    wire[31:0] readData_WB;
    wire[31:0] upperImmediate_WB;
    wire[4:0] regDstMuxOut_WB;
    
    //Write Back***********************************
    wire[31:0] memToRegMuxOut;    // Output of mux that chose between ALUResult(0) or REadData(1)
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //Instruction Fetch**********************************
    mux2x1#(32) IF_Mux(PCAddResult, branchAddress_EX, PCSrc_EX,
                                                            PCSrcOut);
    programcounter IF_ProgramCounter(CLK, PCSrcOut, RST, stall, 
                                                        PCResult);
    pcadder IF_PCAdder(CLK, PCResult,
                                PCAddResult);
    instrMem IF_InstructionMemory(PCResult[10:2], 
                                                    Instruction);
    
    //IF/ID Pipeline Register****************************
    PipelineRegIF_ID IF_ID(CLK, RST, stall, PCAddResult, Instruction, 
                                                        PCAddResult_ID, Instruction_ID);
    
    //Instruction Decode*********************************
    controller ID_Controller(Instruction_ID, 
                                             Controller);
                                            
    mux2x1#(25) ID_CtrlStallMux(Controller, 25'b0, ctrlMuxStall,
                                                                    ControllerMuxOut);
                                                                    
    hazardcontrol ID_HazCtrlUnit(Instruction_ID[25:21], Instruction_ID[20:16], Controller[24], 
                                 PCSrc_EX, regDstOut, memRead_EX, regDstMuxOut_MEM, memRead_MEM,
                                                                                                        ctrlMuxStall, stall);                                                                   
                                                                    
    registerfile ID_RegisterFile(CLK, Instruction_ID[25:21], Instruction_ID[20:16], regDstMuxOut_WB, 
                                     memToRegMuxOut, regWrite_WB,  
                                                                                        readData1, readData2);   
   
    mux2x1#(32) ID_rsFwdMux(readData1, EXEResult_MEM, rsMuxCtrl_ID,
                                                                    rsFwdMuxOut_ID);    
                                                                                                                                                                                                                 
    mux2x1#(32) ID_rtFwdMux(readData2, EXEResult_MEM, rtMuxCtrl_ID,
                                                                    rtFwdMuxOut_ID);
    
    branchmodule ID_BranchModule(rsFwdMuxOut_ID, rtFwdMuxOut_ID, Instruction_ID, PCAddResult_ID, ControllerMuxOut[24:20],
                                                                                                         PCSrc, branchAddress, jal);                                                  
                                                            
    mux2x1#(32) ID_JalDataMux(rsFwdMuxOut_ID, PCAddResult_ID, jal, 
                                                            readData1_jal);
    
    signextend ID_SignExtension(Instruction_ID[15:0], 
                                                        signExtend);
    lowerhalfextend ID_UpperImmediate(Instruction_ID[15:0], 
                                                            upperImmediate);
    
    mux2x1#(5) ID_JalRegMux(Instruction_ID[20:16], 5'b11111, jal,
                                                                    rt_jal);
    
    
    //ID/EX Pipeline Register****************************
    PipeLineRegID_EX ID_EX(CLK, RST, 
                            ControllerMuxOut[1:0], Instruction_ID[5:0],
                            ControllerMuxOut[19], ControllerMuxOut[18:17], ControllerMuxOut[16], ControllerMuxOut[15], ControllerMuxOut[14], ControllerMuxOut[13], ControllerMuxOut[12], 
                            ControllerMuxOut[11], ControllerMuxOut[10], ControllerMuxOut[9:8], ControllerMuxOut[7:3], ControllerMuxOut[2],
                            PCSrc, branchAddress,
                            readData1_jal, rtFwdMuxOut_ID, signExtend, upperImmediate, Instruction_ID[25:21], rt_jal, Instruction_ID[15:11],
                            
                            movCtrl_EX, funct_EX,
                            regWrite_EX, memToReg_EX, memWrite_EX, memRead_EX, ALUSrcA_EX, ALUSrcB_EX, 
                            ALUOutMux_EX, signMux_EX, hiLoMux_EX, DataSelWrite_EX, ALUControl_EX, regDst_EX, 
                            PCSrc_EX, branchAddress_EX,
                            readData1_EX, readData2_EX, immediate_EX, upperImmediate_EX, rs_EX, rt_EX, rd_EX
                            );
    
    //Execution******************************************
    Mux32Bit4To1 EX_rsFwdMux(readData1_EX, EXEResult_MEM, memToRegMuxOut, 32'b0, rsMuxCtrl_EX,
                                                                                                rsFwdMuxOut_EX);
                                                                    
    Mux32Bit4To1 EX_rtFwdMux(readData2_EX, EXEResult_MEM, memToRegMuxOut, 32'b0, rtMuxCtrl_EX,
                                                                                                rtFwdMuxOut_EX);
    
    
    mux2x1#(32) EX_ALUSrcA_Mux(rsFwdMuxOut_EX, rtFwdMuxOut_EX, ALUSrcA_EX, 
                                                                    ALUSrcAMuxOut);
    mux2x1#(32) EX_ALUSrcB_Mux(rtFwdMuxOut_EX, immediate_EX, ALUSrcB_EX, 
                                                                    ALUSrcBMuxOut);
    
    alu#(32) EX_ALU(ALUSrcAMuxOut, ALUSrcBMuxOut, HiLoOut, 
                    ALUControl_EX, 
                                    thicc, ALUResult);
                        
    HiLoReg EX_HiLo(CLK, thicc, 
                                HiLoOut); 
    
    mux2x1#(32) EX_HiLoMux(HiLoOut[31:0], HiLoOut[63:32], hiLoMux_EX, 
                                                                    hiloMuxOut); 
    
    mux2x1#(32) EX_ExeDataMux(ALUResult, hiloMuxOut, ALUOutMux_EX, 
                                                                ALUOutMuxOut);
    
    signextend EX_RegExtendHalf(rtFwdMuxOut_EX[15:0], 
                                                    regDataSEH);
    SignExtendByte EX_RegExtendByte(rtFwdMuxOut_EX[7:0],  
                                                    regDataSEB);
    
    Mux32Bit4To1 EX_DataSel(rtFwdMuxOut_EX, regDataSEH,regDataSEB, immediate_EX, 
                              DataSelWrite_EX, 
                                                dataWriteMemMux);
                                  
    mux2x1#(5) EX_RegDestMux(rt_EX, rd_EX, regDst_EX, 
                                                    regDstOut);
    
    
    MovComponent EX_MovComponent(rtFwdMuxOut_EX, funct_EX, movCtrl_EX, 
                                                              isMov);
                                                              
    mux2x1#(1) EX_MovMux(regWrite_EX, 1'b1, isMov,     
                                                    movMux);
                                                    
    ForwardingUnit EX_FwdUnit(rs_EX, rt_EX, Instruction_ID[25:21], Instruction_ID[20:16], ControllerMuxOut[24], regWrite_MEM, regDstMuxOut_MEM, regWrite_WB, regDstMuxOut_WB,
                                                                                                                  rsMuxCtrl_EX, rtMuxCtrl_EX, rsMuxCtrl_ID, rtMuxCtrl_ID);
    
    //EX/MEM Pipeline Register***************************
    PipelineRegEX_MEM EX_MEM(CLK, RST, 
                             movMux, memToReg_EX, memWrite_EX, memRead_EX, DataSelWrite_EX,
                             ALUOutMuxOut, dataWriteMemMux, upperImmediate_EX, regDstOut,
                             
                             regWrite_MEM, memToReg_MEM, memWrite_MEM, memRead_MEM, DataSelWrite_MEM,
                             EXEResult_MEM, dataWriteMem_MEM, upperImmediate_MEM, regDstMuxOut_MEM);
                                        
    //Memory Access**************************************
    DataMemory MEM_DataMemory(EXEResult_MEM, dataWriteMem_MEM, CLK, memWrite_MEM, memRead_MEM, readData);
    
    signextend MEM_MemExtendHalf(readData[15:0], 
                                                    memDataSEH);
    SignExtendByte MEM_MemExtendByte(readData[7:0],  
                                                    memDataSEB);
    
    Mux32Bit4To1 MEM_DataSel(readData, memDataSEH, memDataSEB, 32'b0, 
                                DataSelWrite_MEM, 
                                                    readDataMux); 
    
    //MEM/WB Pipeline Register***************************
    PipelineRegMEM_WB MEM_WB(   CLK, RST,
                                regWrite_MEM, memToReg_MEM,
                                readDataMux, EXEResult_MEM, upperImmediate_MEM, regDstMuxOut_MEM,
                                
                                regWrite_WB, memToReg_WB,
                                readData_WB, EXEResult_WB, upperImmediate_WB, regDstMuxOut_WB);
        
    //Write Back*****************************************
    Mux32Bit4To1 WB_MemToRegMux(EXEResult_WB, readData_WB, upperImmediate_WB, 32'b0, 
                                memToReg_WB, 
                                            memToRegMuxOut); 
    
    assign REGADDR = regDstMuxOut_WB;
    assign OUTVAL = memToRegMuxOut;
    assign HI = HiLoOut[63:32];
    assign LO = HiLoOut[31:0];
    assign REGWRITE = regWrite_WB;
    assign A = ALUSrcAMuxOut;
    assign B = ALUSrcBMuxOut;

endmodule
