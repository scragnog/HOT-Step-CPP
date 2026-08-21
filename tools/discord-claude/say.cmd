@echo off
rem Interject into the working-group thread from this PC.
rem   say                      -> contextual interjection, bot's choice
rem   say "look at the v5 CE"  -> steered interjection
setlocal
set NOTE=%~1
curl -s -X POST http://127.0.0.1:47821/interject -H "Content-Type: application/json" -d "{\"note\": \"%NOTE%\"}"
echo.
