

            $Host.UI.RawUI.WindowTitle = 'Clipppy - CASEOH_'
            $ErrorActionPreference = 'Stop'
            $ProgressPreference = 'SilentlyContinue'
            
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
            

# Error handling
$ErrorActionPreference = 'Stop'
try {
    cd 'C:\Users\samfi\OneDrive\Desktop\cursor projects\clipppy 2'
    python twitch_clip_bot.py start --streamer caseoh_ --always-on-mode
} catch {
    Write-Error $_.Exception.Message
    Write-Error $_.ScriptStackTrace
    Start-Sleep -Seconds 30  # Keep window open to see error
    throw
}
