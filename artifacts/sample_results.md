# Sample results

| file | transcript | intent | confidence | response | expected | actual | match |
|---|---|---|---|---|---|---|---|
| 01_weather.wav | What's the weather like today? | weather | 0.987 | I can check the weather for you — in a real deployment this would call a weather API for your location. | weather | weather | Y |
| 02_time.wav | What time is it right now? | time | 1.000 | It's currently 06:32 PM (Asia/Kolkata). | time | time | Y |
| 03_tell_joke.wav | Tell me a joke. | tell_joke | 1.000 | Why did the developer go broke? Because they used up all their cache. | tell_joke | tell_joke | Y |
| 04_book_flight.wav | Book of Light from Chennai to Delhi | oos | 0.229 | Sorry, I don't have an answer for that. I'm best at things like telling the time, checking flight status, or answering banking questions. | book_flight | oos | N |
| 05_balance.wav | What's my banker on the balance? | balance | 0.667 | I can look that up — a live version would fetch your balance directly. | balance | balance | Y |
| 06_alarm.wav | Set an alarm for 6.30 tomorrow morning. | alarm | 0.997 | I can start that alarm — a live version would set it and notify you at the right time. | alarm | alarm | Y |
| 07_translate.wav | How do you say thank you in French? | translate | 0.999 | I can help with translation — a live version would return the translated text. | translate | translate | Y |
| 08_flip_coin.wav | Can you flip a coin for me? | flip_coin | 1.000 | I flipped a coin: heads! | flip_coin | flip_coin | Y |
| 09_oos.wav | Who won the Cricket World Cup in 2011? | oos | 0.733 | That's outside what I can do right now. I can help with tasks like setting a timer, checking your account balance, or getting the weather. | oos | oos | Y |
| 10_silence.wav |  | oos | 0.000 | I didn't catch that — the audio seemed to be silence. Please try again and speak clearly into the microphone. | silence | silent | Y |

**9/10 matched**
