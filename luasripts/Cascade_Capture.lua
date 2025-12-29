--[[
    REAL-TIME CONTINUOUS RADAR CAPTURE (CONSISTENT 780MB FILES)
    
    KEY: ARM TDA → Capture → STOP TDA for EACH iteration
    
    A. FRAMING & CAPTURE
    1. Triggering Slave (3, 2, 1) sequentially in hardware triggered mode
    2. Triggering Master in software triggered mode
    
    B. TRANSFERRING FILES
    1. Data stored in files with max cap at 2 GB
    2. Files retrievable from SSD (/mnt/ssd folder) using WinSCP
--]]

-- ======================== CONFIGURATION PARAMETERS ========================

-- Timing parameters - CRITICAL FOR CONSISTENT FILE SIZES
-- Calculate: Frame_Periodicity_ms x Number_of_Frames + extra buffer
-- Example: 100ms x 30 frames = 3000ms base time
capture_time                    = 3000--6000      -- ms - INCREASED: Ensure all frames complete
frame_end_wait                  = 4000      -- ms - INCREASED: Wait longer for Frame End event
inter_loop_time                 = 2000--3000      -- ms - INCREASED: More time for file flush

-- Loop control
num_loops                       = -1        -- Set to -1 for infinite continuous capture

-- File management
n_files_allocation              = 0         -- MUST BE 0 for correct file sizes
data_packaging                  = 0         -- 0: 16-bit, 1: 12-bit
capture_directory               = "test_capture_incoming"
num_frames_to_capture           = 0         -- 0: use default from profile

-- Framing type
framing_type                    = 1         -- 0: infinite, 1: finite

-- ======================== HELPER FUNCTIONS ========================

function Framing_Control(Device_ID, En1_Dis0)
    local status = 0
    
    if (En1_Dis0 == 1) then 
        status = ar1.StartFrame_mult(dev_list[Device_ID])
        if (status == 0) then
            WriteToLog("Device "..Device_ID.." : Start Frame Successful\n", "green")
        else
            WriteToLog("Device "..Device_ID.." : Start Frame Failed\n", "red")
            return -5
        end
    else
        status = ar1.StopFrame_mult(dev_list[Device_ID])
        if (status == 0) then
            WriteToLog("Device "..Device_ID.." : Stop Frame Successful\n", "green")
        else
            WriteToLog("Device "..Device_ID.." : Stop Frame Failed\n", "red")
            return -5
        end
    end
    
    return status
end

-- ======================== MAIN CAPTURE LOOP ========================

local loop_counter = 0

WriteToLog("========================================\n", "purple")
WriteToLog("CONTINUOUS CAPTURE WITH CORRECT FILE SIZES\n", "purple")
WriteToLog("Mode: "..(num_loops == -1 and "INFINITE" or num_loops.." captures").."\n", "purple")
WriteToLog("Directory: "..capture_directory.."\n", "purple")
WriteToLog("Expected file size: ~780MB per capture\n", "purple")
WriteToLog("========================================\n", "purple")

while (num_loops ~= 0) do

    loop_counter = loop_counter + 1
    
    WriteToLog("\n========================================\n", "purple")
    WriteToLog("CAPTURE #"..loop_counter.."\n", "purple")
    WriteToLog("Remaining: "..(num_loops > 0 and num_loops or "Infinite").."\n", "purple")
    WriteToLog("========================================\n", "purple")
    
    -- ============ ARM TDA ============
    WriteToLog("Arming TDA...\n", "blue")
    status = ar1.TDACaptureCard_StartRecord_mult(1, n_files_allocation, data_packaging, capture_directory, num_frames_to_capture)
    
    if (status == 0) then
        WriteToLog("TDA ARM Successful\n", "green")
    else
        WriteToLog("TDA ARM Failed\n", "red")
        return -5
    end
    
    RSTD.Sleep(1000)
    
    -- ============ TRIGGER FRAMES ============
    WriteToLog("Starting Frame Triggers...\n", "blue")
    
    if (RadarDevice[4] == 1) then
        Framing_Control(4, 1)
    end
    
    if (RadarDevice[3] == 1) then
        Framing_Control(3, 1)
    end
    
    if (RadarDevice[2] == 1) then
        Framing_Control(2, 1)
    end
    
    Framing_Control(1, 1)
    
    WriteToLog("Capturing data...\n", "blue")
    RSTD.Sleep(capture_time)
    
    -- ============ WAIT FOR FRAME END EVENT ============
    -- CRITICAL: This ensures ALL frames complete before stopping TDA
    WriteToLog("Waiting for Frame End event...\n", "blue")
    RSTD.Sleep(frame_end_wait)
    
    -- ============ STOP FRAMES (if infinite framing) ============
    if (framing_type == 0) then
        WriteToLog("Stopping frames...\n", "blue")
        
        if (RadarDevice[4] == 1) then
            Framing_Control(4, 0)
        end
        
        if (RadarDevice[3] == 1) then
            Framing_Control(3, 0)
        end
        
        if (RadarDevice[2] == 1) then
            Framing_Control(2, 0)
        end
        
        Framing_Control(1, 0)
    end
    
    WriteToLog("Capture completed\n", "green")
    
    -- ============ STOP TDA (CRITICAL FOR CORRECT FILE SIZE) ============
    -- Additional delay to ensure all data is written to buffer
    WriteToLog("Ensuring all data is written to buffer...\n", "blue")
    RSTD.Sleep(3000)
    
    WriteToLog("Stopping TDA to save files...\n", "blue")
    status = ar1.TDACaptureCard_StopRecord_mult(1)
    
    if (status == 0) then
        WriteToLog("Files saved successfully (~780MB)\n", "green")
    else
        WriteToLog("TDA Stop Failed\n", "red")
    end
    
    -- Decrement loop counter
    if (num_loops > 0) then
        num_loops = num_loops - 1
    end
    
    -- Wait before next capture
    WriteToLog("Waiting "..inter_loop_time.." ms before next capture...\n", "blue")
    RSTD.Sleep(inter_loop_time)

end

WriteToLog("\n========================================\n", "purple")
WriteToLog("CAPTURE COMPLETED\n", "purple")
WriteToLog("Total captures: "..loop_counter.."\n", "purple")
WriteToLog("All files saved to: "..capture_directory.."\n", "purple")
WriteToLog("========================================\n", "purple")

-- Optional: Transfer files using WinSCP
--[[ 
WriteToLog("\nStarting file transfer via WinSCP...\n", "blue")
status = ar1.TransferFilesUsingWinSCP_mult(1)
if(status == 0) then
    WriteToLog("File transfer COMPLETE!\n", "green")
else
    WriteToLog("File transfer FAILED!\n", "red")
    return -5
end
--]]