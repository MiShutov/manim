# Русская озвучка с F5-TTS (v4_winter)

Локальная генерация русской речи с клонированием голоса. Модель — финетюн
`F5TTS_v1_Base_v4_winter` (`models/F5TTS_v1_Base_v4_winter/`), инференс через
`f5-tts` 1.1.22.

Код и документация — в `text2speech/`, медиафайлы — в `audio/text2speech/`.
Код, комментарии и сообщения скриптов — на английском.

## Быстрый запуск

```bash
cd text2speech
./run_tts.sh                     # голос по умолчанию (winter)
./run_tts.sh --voice nastya_street
./run_tts.sh --list              # список голосов

./run_single_text.sh /path/to/text.txt winter   # один файл -> output/text/winter.wav
./add_voice.sh /path/to/voice.mp4 --name anna   # новый голос из аудио/видео
```

Результат: `audio/text2speech/output/<text>/<voice>.wav` (24 kHz, mono,
PCM 16 bit) и `<voice>.stressed.txt` — текст ровно в том виде, в котором он
ушёл в модель.

Ударения расставляются автоматически, вручную их править не обязательно.

## Голоса

Голос — это папка в `audio/text2speech/ref_voices/`:

```
ref_voices/
  winter/
    ref.wav   # 10.45 s, 24 kHz, mono
    ref.txt   # точная расшифровка ref.wav
```

Скрипт выбирает папку по `--voice NAME` (или `VOICE=NAME`). Если флаг не
задан, берётся `winter`; если её нет — единственный доступный голос.

Новый голос из любого аудио или видео (mp3, m4a, mp4, wav, ...):

```bash
./add_voice.py ../audio/text2speech/ref_voices/some_recording.mp4
# -> ref_voices/some_recording/ref.wav + ref.txt
```

Что делает `add_voice.py`:

1. вытаскивает из файла моно-дорожку 24 kHz;
2. если запись длиннее 12 с, находит границу слова около 10.5 с по таймингам
   локального Qwen3-ASR + forced aligner и вырезает фрагмент;
3. распознаёт уже сам фрагмент и кладёт текст в `ref.txt` — текст всегда
   соответствует аудио.

Требования к референсу: 8–12 с чистой речи без музыки, 24 kHz mono.

Длинный референс целиком:

```bash
./add_voice.sh ../audio/text2speech/ref_voices/reference_voice.wav \
  --name winter_full --max-seconds 0
```

`--max-seconds 0` отключает обрезку: в `ref.wav` попадёт вся запись, а в
`ref.txt` — её полная расшифровка. `synthesize.py` не обрезает референс перед
инференсом, поэтому модель получает весь кусок. Это работает, но финетюн
обучен на промптах 8–12 с: длинный референс замедляет генерацию и может
снизить стабильность голоса.
Ошибки распознавания в `ref.txt` можно и нужно править руками: транскрипт
влияет на качество клонирования.

## Ударения

Модель натренирована на тексте, где ударение помечено `+` перед ударной
гласной: `с+интеза`, `нейр+онной`, `сет+и`. `add_stress.py` (библиотека
Silero stress) расставляет такие метки автоматически:

```bash
./add_stress.py ../audio/text2speech/input_text.txt /tmp/stressed.txt --dry-run
```

Правила:

- ручные `+` в `input_text.txt` сохраняются и не переставляются;
- односложные слова не размечаются — там ударение однозначно;
- `ё` не размечается, она всегда ударная;
- омографы (`за́мок` / `замо́к`) решаются по контексту, но не идеально —
  спорные слова проверяйте в `output/<text>/<voice>.stressed.txt`.

Отключить автоматику: `AUTO_STRESS=0 ./run_tts.sh`.

Почему именно `+`: попытка заменить его на юникодный combining acute
`U+0301` (стандартная запись ударения, этот символ тоже есть в словаре)
ухудшала результат — модель этот знак как ударение не воспринимает.

Если ударение не то: сдвиньте `+` вручную (`родн+ое` -> `р+одное`),
используйте `ё`, где это уместно, или перепишите заимствование так, как оно
звучит (`интэрнэт`).

## Что можно задавать

| Переменная | По умолчанию | Действие |
| --- | --- | --- |
| `VOICE` | `winter` | имя папки голоса, альтернатива `--voice` |
| `TEXT_FILE` | `input_text.txt` | входной текст, альтернатива `--text` |
| `OUTPUT_FILE` | `<voice>.wav` | имя файла внутри `output/<text>/` |
| `AUDIO_DIR` | `<repo>/audio/text2speech` | папка с входными и выходными файлами |
| `AUTO_STRESS` | `1` | автоматическая расстановка `+` |
| `CONDA_ENV` | `f5tts` | conda-окружение для запуска |
| `DEVICE` | `cuda` | `cuda` или `cpu` |
| `NFE_STEP` | `32` | шаги диффузии: 16 быстрее, 64 чище |
| `CFG_STRENGTH` | `2.0` | сила следования тексту |
| `SPEED` | `1.0` | темп речи |
| `SWAY` | — | `sway_sampling_coef`, например `0.5` |
| `FIX_DURATION` | — | фиксированная длительность (референс + генерация), сек |
| `TARGET_RMS` | — | громкость выходного сигнала |
| `REMOVE_SILENCE` | — | вырезать длинные паузы из результата |
| `MODEL_DIR` | `<repo>/models/F5TTS_v1_Base_v4_winter` | папка модели |
| `PYTHON` | из окружения | явный путь к интерпретатору |
| `SEED` | — | фиксированный seed для воспроизводимости |

Примеры:

```bash
./run_tts.sh --voice nastya_street_2
NFE_STEP=64 SPEED=0.95 ./run_tts.sh
AUTO_STRESS=0 ./run_tts.sh                    # текст без разметки, как есть
SWAY=0.5 FIX_DURATION=14 ./run_tts.sh
REMOVE_SILENCE=1 ./run_tts.sh
```

Что задаётся текстом, а не флагами:

- пунктуация: запятые, точки и тире задают паузы и интонацию;
- числа и аббревиатуры лучше писать словами — нормализации чисел в модели
  нет, цифры читаются посимвольно;
- длинный текст автоматически режется на куски примерно по 130 символов
  (порог считается от длины референса) и склеивается кроссфейдом.

Чего нет в CLI: фиксированного seed (генерация стохастическая; для
воспроизводимости нужен Python API `F5TTS.infer(..., seed=...)`).

## Раскладка файлов

Код — `text2speech/`:

| Файл | Назначение |
| --- | --- |
| `run_tts.sh` | запуск генерации, выбор голоса |
| `run_single_text.sh` | один текстовый файл + голос |
| `add_voice.sh` | короткая обёртка над `add_voice.py` |
| `add_stress.py` | автоматическая расстановка `+` |
| `add_voice.py` | подготовка папки голоса из аудио/видео |
| `synthesize.py` | инференс F5-TTS: локальные веса и вокодер |
| `env_tools.py` | автоперезапуск скриптов в окружении `f5tts` |
| `requirements.txt` | python-зависимости |
| `README.md` | эта документация |

Модель — `models/F5TTS_v1_Base_v4_winter/`:

| Файл | Назначение |
| --- | --- |
| `model_212000.safetensors` | веса финетюна |
| `vocab.txt` | словарь символьного токенизатора (2545 символов) |
| `model_cfg.yaml` | конфиг архитектуры для инференса |

Вокодер — `models/vocos-mel-24khz/` (`config.yaml`, `pytorch_model.bin`,
54 МБ). Скачивается один раз:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

target = Path("models/vocos-mel-24khz")
target.mkdir(parents=True, exist_ok=True)
for name in ("config.yaml", "pytorch_model.bin"):
    shutil.copy2(hf_hub_download("charactr/vocos-mel-24khz", name), target / name)
PY
```

Данные — `audio/text2speech/`:

| Путь | Назначение |
| --- | --- |
| `input_text.txt` | текст для синтеза (UTF-8), `+` можно ставить вручную |
| `ref_voices/<voice>/` | голоса: `ref.wav` + `ref.txt` |
| `output/<text>/` | `<voice>.wav` и `<voice>.stressed.txt` |
| `reference_voice.wav` | исходная запись голоса winter, 30 с |
| `reference_voice.words.json` | слова и тайминги исходной записи |

## Конфиг модели

Скачанный `F5TTS_v1_Base_v4_winter.yaml` побайтово совпадал с официальным
`F5TTS_v1_Base.yaml` из пакета `f5-tts`, и веса ему соответствуют: 22 блока
DiT, `dim=1024`, `text_dim=512`, `mel=100`, Vocos 24 kHz,
`input_embed.proj` имеет вход `2*100+512=712`, `text_embed` — 2546 строк
(2545 символов словаря + filler).

Единственное, что в нём неверно для русского финетюна, — метаданные
токенизатора: там `tokenizer: pinyin` и `datasets.name: Emilia_ZH_EN`.
Инференс эти поля не читает (CLI всегда берёт словарь из `--vocab_file` и
работает в режиме `custom`), поэтому рабочий конфиг лежит рядом с весами:
`models/F5TTS_v1_Base_v4_winter/model_cfg.yaml`. В нём оставлена только
секция `model` (её читает инференс), указано `tokenizer: custom` и
`tokenizer_path: vocab.txt`.

## Проверка результата

Круговой тест: распознать выходной файл тем же ASR и сравнить с исходным
текстом.

```bash
python ../speech2text/transcribe.py \
  ../audio/text2speech/output/input_text/my_normal.wav -o /tmp/speech_check.json
```

Ударения ASR не показывает, их нужно проверять на слух.

## Замечания

- Vocoder Vocos лежит локально в `models/vocos-mel-24khz`, интернет при
  генерации не нужен: `synthesize.py` загружает вокодер из
  `models/vocos-mel-24khz`.
- Модель ударений Silero идёт внутри пакета `silero-stress`, ничего не
  скачивает.
- `nfe_step` — компромисс скорость/качество: 16 быстрее, 32 чище, 64 лучше.
