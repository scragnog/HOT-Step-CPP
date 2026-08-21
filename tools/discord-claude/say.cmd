@echo off
rem Interject into the working-group thread from this PC.
rem   say                      -> contextual interjection, bot's choice
rem   say "look at the v5 CE"  -> steered interjection
setlocal
set NOTE=%~1
curl -sS --max-time 300 --fail-with-body -X POST http://127.0.0.1:47821/interject -H "Content-Type: application/json" -d "{\"note\": \"%NOTE%\"}"
if errorlevel 1 (
  echo.
  echo [say] FAILED — is the bot running? Start it with:  npm start  in tools\discord-claude
  echo [say] ^(the interject endpoint needs the current code — restart the bot if it predates it^)
)
echo.
