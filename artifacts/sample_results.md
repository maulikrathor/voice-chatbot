# Sample results

| file | transcript | intent | confidence | response | expected | actual | match |
|---|---|---|---|---|---|---|---|
| 01_weather.wav | What's the weather like today? | weather | 0.987 | I can check the weather for you — in a real deployment this would call a weather API for your location. | weather | weather | Y |
| 02_time.wav | What time is it right now? | time | 1.000 | It's currently 11:31 PM (Asia/Kolkata). | time | time | Y |
| 03_tell_joke.wav | Tell me your joke. | tell_joke | 0.927 | I tried to write a joke about UDP, but you might not get it. | tell_joke | tell_joke | Y |
| 04_book_flight.wav | Book a flight from Chennai to Delhi. | book_flight | 0.987 | I can help book a flight — in a real deployment this would search fares and reserve a seat. | book_flight | book_flight | Y |
| 05_balance.wav | What's my bank account balance? | balance | 0.948 | I can look that up — a live version would fetch your balance directly. | balance | balance | Y |
| 06_alarm.wav | Set an alarm for 6.30 tomorrow morning. | alarm | 0.997 | I can start that alarm — a live version would set it and notify you at the right time. | alarm | alarm | Y |
| 07_translate.wav | How do you say thank you in French? | translate | 0.999 | I can help with translation — a live version would return the translated text. | translate | translate | Y |
| 08_flip_coin.wav | Can you flip a coin for me? | flip_coin | 1.000 | I flipped a coin: tails! | flip_coin | flip_coin | Y |
| 09_oos.wav | Who won the cricket World Cup in 2011? | oos | 0.733 | I'm not able to help with that. I'm built to handle things like checking your balance, telling you the weather, or booking a flight — try one of those! | oos | oos | Y |
| 10_silence.wav |  | oos | 0.000 | I didn't catch that — the audio seemed to be silence. Please try again and speak clearly into the microphone. | silence | silent | Y |

**10/10 matched**
