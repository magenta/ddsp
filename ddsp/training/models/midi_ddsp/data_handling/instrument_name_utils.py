"""The utilities for handling instrument names, abbreviations, ids and midi
programs used in the URMP dataset."""

INST_NAME_TO_ABB_DICT = {
  'violin': 'vn',
  'viola': 'va',
  'cello': 'vc',
  'double bass': 'db',
  'flute': 'fl',
  'oboe': 'ob',
  'clarinet': 'cl',
  'saxophone': 'sax',
  'bassoon': 'bn',
  'trumpet': 'tpt',
  'horn': 'hn',
  'trombone': 'tbn',
  'tuba': 'tba',
  'guitar': 'gtr'
}

INST_ABB_TO_NAME_DICT = {v: k for k, v in INST_NAME_TO_ABB_DICT.items()}

INST_ABB_LIST = list(INST_NAME_TO_ABB_DICT.values())

INST_NAME_LIST = list(INST_NAME_TO_ABB_DICT.keys())

INST_ABB_TO_ID_DICT = {v: i for i, (k, v) in
                       enumerate(INST_NAME_TO_ABB_DICT.items())}

INST_ID_TO_ABB_DICT = {i: v for i, (k, v) in
                       enumerate(INST_NAME_TO_ABB_DICT.items())}

INST_NAME_TO_ID_DICT = {k: i for i, (k, v) in
                        enumerate(INST_NAME_TO_ABB_DICT.items())}

INST_ID_TO_NAME_DICT = {i: k for i, (k, v) in
                        enumerate(INST_NAME_TO_ABB_DICT.items())}

NUM_INST = 20

INST_NAME_TO_MIDI_PROGRAM_DICT = {
  'violin': 41,
  'viola': 42,
  'cello': 43,
  'double bass': 44,
  'flute': 74,
  'oboe': 69,
  'clarinet': 72,
  'saxophone': 67,
  'bassoon': 71,
  'trumpet': 57,
  'horn': 61,
  'trombone': 58,
  'tuba': 59,
  'guitar': 27
}

INST_ABB_TO_MIDI_PROGRAM_DICT = {INST_NAME_TO_ABB_DICT[k]: v for k, v in
                                 INST_NAME_TO_MIDI_PROGRAM_DICT.items()}

INST_ID_TO_MIDI_PROGRAM_DICT = {INST_ABB_TO_ID_DICT[k]: v for k, v in
                                INST_ABB_TO_MIDI_PROGRAM_DICT.items()}

MIDI_PROGRAM_TO_INST_NAME_DICT = {v: k for k, v in
                                  INST_NAME_TO_MIDI_PROGRAM_DICT.items()}

MIDI_PROGRAM_TO_INST_ABB_DICT = {v: k for k, v in
                                 INST_ABB_TO_MIDI_PROGRAM_DICT.items()}

MIDI_PROGRAM_TO_INST_ID_DICT = {v: k for k, v in
                                INST_ID_TO_MIDI_PROGRAM_DICT.items()}
