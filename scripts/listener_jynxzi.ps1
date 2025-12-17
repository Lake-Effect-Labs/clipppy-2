

            # Set execution policy and error handling
            Set-ExecutionPolicy Bypass -Scope Process -Force
            $ErrorActionPreference = 'Stop'
            $ProgressPreference = 'SilentlyContinue'
            
            # Configure window
            $Host.UI.RawUI.WindowTitle = 'Clipppy - JYNXZI'
            
            # Prevent sleep/hibernate for this session
            Add-Type -TypeDefinition @"
                using System;
                using System.Runtime.InteropServices;
                public class DisplayState {
                    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
                    public static extern uint SetThreadExecutionState(uint esFlags);
                    public const uint ES_CONTINUOUS = 0x80000000;
                    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
                    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
                }
"@
            [DisplayState]::SetThreadExecutionState([DisplayState]::ES_CONTINUOUS -bor [DisplayState]::ES_SYSTEM_REQUIRED -bor [DisplayState]::ES_DISPLAY_REQUIRED)
            
            # Set process priority
            $process = Get-Process -Id $pid
            $process.PriorityClass = 'High'
            
            # Change to working directory
            cd 'C:\Users\samfi\OneDrive\Desktop\cursor projects\clipppy 2'
            
            # Activate Python environment if it exists
            if (Test-Path "venv/Scripts/activate.ps1") {
                ./venv/Scripts/activate.ps1
            } elseif (Test-Path ".venv/Scripts/activate.ps1") {
                ./.venv/Scripts/activate.ps1
            }
            
            # Set up error handling
            try {
                # Run bot with full error capture
                python twitch_clip_bot.py start --streamer jynxzi --always-on-mode 2>&1 | Tee-Object -FilePath "logs/listener_jynxzi.log"
            } catch {
                Write-Host "❌ Error running bot: $_"
                Write-Host $_.ScriptStackTrace
                Start-Sleep -Seconds 30  # Keep window open to see error
                throw
            }
            

# Error handling
$ErrorActionPreference = 'Stop'
try {
    cd 'C:\Users\samfi\OneDrive\Desktop\cursor projects\clipppy 2'
    python twitch_clip_bot.py start --streamer jynxzi --always-on-mode
} catch {
    Write-Error $_.Exception.Message
    Write-Error $_.ScriptStackTrace
    Start-Sleep -Seconds 30  # Keep window open to see error
    throw
}
