# Round 1: Syntactic Decomposition of CALVIN Text Corpus

**Date:** 2026-03-16
**Goal:** Decompose CALVIN instructions beyond verb extraction — identify objects, colors, locations, directions, and syntactic templates.

## Corpus Overview

- **259 unique instructions**, **3,461 total samples** (train split)
- All instructions are imperative sentences (robot commands)
- spaCy `en_core_web_sm` used for POS tagging and dependency parsing
- **22 verb classes** used for classification (see `data/verb_classes.txt`)

## Semantic Slot Schema

Each instruction can be decomposed into up to 6 slots:

| Slot | Description | Example |
|------|-------------|---------|
| **verb** | Action verb; includes particle when the verb+particle is a class (`pick up`, `take off`, `turn on`, `turn off`) | `push`, `pick up`, `turn on` |
| **object** | Direct object being manipulated | `block`, `door`, `switch` |
| **object_modifier** | Adjective/participle modifying the object: color (`blue`, `red`, `pink`), state (`grasped`, `stacked`, `top`), type (`sliding`, `led`) | `blue`, `grasped`, `sliding` |
| **source_location** | Where object currently is | `drawer`, `shelf`, `table` |
| **target_location** | Destination for object | `drawer`, `cabinet`, `top of stack` |
| **direction** | Spatial direction of action | `left`, `right`, `up`, `down` |

### Particle resolution rule

spaCy extracts verb particles (e.g. `up`, `off`, `down`) as part of phrasal verbs. We resolve them as follows:
- If verb+particle matches a verb class → treat as a single verb (e.g. `pick` + `up` → verb=`pick up`)
- If verb+particle does NOT match a verb class → particle is a direction (e.g. `lift` + `up` → verb=`lift`, direction=`up`; `slide` + `down` → verb=`slide`, direction=`down`)

Verb+particle combos that are verb classes: **pick up**, **take off**, **turn on**, **turn off**
Verb+particle combos collapsed to base verb (particle → direction): **lift up** → `lift` + `up`, **slide up/down** → `slide` + `up`/`down`, **move up** → `move` + `up`

### Slot Fill Rates (259 unique instructions)

| Slot | Filled | Rate |
|------|--------|------|
| verb | 259 | 100.0% |
| object | 239 | 92.3% |
| object_modifier | 209 | 80.7% |
| direction | 89 | 34.4% |
| target_location | 81 | 31.3% |
| source_location | 30 | 11.6% |

**Observation:** Nearly all instructions name an object (92%) and include a modifier (81%). Directions and locations are optional.

---

## 1. Verbs

### 22 verb classes (from `data/verb_classes.txt`)

Counted over 259 unique instructions:

| Verb | Count | | Verb | Count |
|------|-------|-|------|-------|
| pick up | 35 | | turn off | 8 |
| push | 34 | | sweep | 8 |
| slide | 26 | | store | 4 |
| grasp | 24 | | stack | 4 |
| rotate | 21 | | take off | 3 |
| lift | 19 | | open | 3 |
| place | 15 | | close | 2 |
| put | 15 | | pull | 2 |
| turn | 14 | | remove | 2 |
| move | 11 | | collapse | 1 |
| turn on | 8 | | unstack | 1 |

**Key patterns:**
- Synonym clusters: {`pick up`, `grasp`, `lift`}, {`place`, `put`, `store`}, {`push`, `slide`, `sweep`}
- `turn` is ambiguous: `turn [block]` (rotate) vs `turn on/off [light]` (switch)

---

## 2. Objects

### Object Types (sample-level, N=3,461)

| Object Type | Samples | % |
|-------------|---------|---|
| block | 2,522 | 72.9% |
| drawer | 345 | 10.0% |
| door/slider | 232 | 6.7% |
| light | 205 | 5.9% |
| switch | 77 | 2.2% |
| object (generic) | 72 | 2.1% |
| other | 8 | 0.2% |

**Block dominance:** 73% of all samples involve blocks. The remaining 27% are fixture interactions (drawer, door, lights, switch).

### Direct Objects (unique instructions)

| Object | Count | | Object | Count |
|--------|-------|-|--------|-------|
| block | 160 | | lamp | 5 |
| object | 25 | | light | 5 |
| door | 22 | | blocks | 4 |
| it | 13 | | handle | 2 |
| drawer | 6 | | led | 2 |
| switch | 6 | | bulb | 1 |
| degrees | 6 | | | |

**Note:** "object" is a generic placeholder in `place/put/store the [grasped] object` instructions, where the specific block color was already grasped. "it" appears in compound instructions like "go towards the block and pick **it** up."

---

## 3. Object Modifiers

### By type (unique instructions)

| Type | Modifier | Count | Modifies | Example |
|------|----------|-------|----------|---------|
| color | blue | 52 | block | "push the **blue** block" |
| color | red | 51 | block | "lift the **red** block" |
| color | pink | 50 | block | "grasp the **pink** block" |
| color | yellow | 4 | light/lamp | "turn off the **yellow** lamp" |
| color | green | 4 | light/lamp | "turn on the **green** light" |
| state | grasped | 11 | object/block | "place the **grasped** object in the drawer" |
| state | stacked | 2 | block/blocks | "take off the **stacked** block" |
| state | top | 2 | block/one | "remove the **top** block" |
| type | sliding | 18 | cabinet/door | "pick up the block from the **sliding** cabinet" |
| type | led | 4 | lamp/light | "turn on the **led** lamp" |
| type | light | 2 | bulb | "turn off the **light** bulb" |

### Sample-level color distribution

| Color | Samples | % of total | Context |
|-------|---------|------------|---------|
| blue | 706 | 20.4% | blocks only |
| red | 710 | 20.5% | blocks only |
| pink | 683 | 19.7% | blocks only |
| yellow | 68 | 2.0% | lights only |
| green | 28 | 0.8% | lights only |

- Block colors (blue/red/pink) are roughly balanced (~20% each)
- Light colors (yellow/green) are rare — only in `turn on/off` instructions
- **60.6% of samples mention a color**; the rest are fixture actions or generic object references
- Non-color modifiers (`grasped`, `stacked`, `top`, `sliding`, `led`) appear in ~20% of unique instructions, primarily indicating object state or fixture type

---

## 4. Locations

### Target Locations (where object goes)

| Location | Unique Instr | Context |
|----------|-------------|---------|
| drawer | 35 | `place/put/store/push X into drawer` |
| cabinet | 19 | `place/put X in cabinet` |
| slider | 10 | `place/put X in slider` |
| top of stack | 8 | `stack/place X on top of another block` |
| shelf | 6 | `place X on shelf` |
| table | 3 | `place X on table` |

### Source Locations (where object is)

| Location | Unique Instr | Context |
|----------|-------------|---------|
| drawer | 9 | `pick up X from/in drawer` |
| cabinet | 9 | `grasp X lying in cabinet` |
| table | 6 | `lift X from table` |
| slider | 3 | `pick up X from slider` |
| shelf | 3 | `lift X on shelf` |

### Location mention rates (sample-level)
- 59.7% of samples mention **no** location
- 35.9% mention **1** location
- 4.4% mention **2** locations (source + target, e.g., "pick up the block from the drawer" where both drawer context and implicit table origin)

---

## 5. Directions

### Sample-level distribution

| Direction | Samples | % |
|-----------|---------|---|
| (none) | 1,797 | 51.9% |
| left | 651 | 18.8% |
| right | 581 | 16.8% |
| on top | 165 | 4.8% |
| into | 150 | 4.3% |
| down | 40 | 1.2% |
| up | 37 | 1.1% |
| off (spatial) | 40 | 1.2% |

- Left/right directions appear with `push`, `slide`, `sweep`, `move`, `rotate` + block/door
- Up/down only with `switch` and `move`/`slide` (the switch)
- "into" = pushing/sweeping block into drawer

---

## 6. Verb x Object Cross-Tabulation (sample-level)

| Verb | block | door/slider | drawer | light | switch | object(generic) |
|------|-------|-------------|--------|-------|--------|-----------------|
| push | 346 | 60 | 90 | — | 32 | — |
| pick up | 428 | — | — | — | — | — |
| slide | 317 | 51 | 31 | — | 22 | — |
| rotate | 334 | — | — | — | — | — |
| grasp | 282 | — | — | — | — | — |
| lift | 217 | — | — | — | — | — |
| place | 75 | 18 | 31 | — | — | 33 |
| sweep | 121 | — | 18 | — | — | — |
| put | 77 | 10 | 17 | — | — | 22 |
| move | — | 93 | — | — | 23 | — |
| turn on | — | — | — | 105 | — | — |
| turn off | — | — | — | 100 | — | — |
| turn | 66 | — | — | — | — | — |
| stack | 56 | — | — | — | — | 17 |
| open | — | — | 43 | — | — | — |
| close | — | — | 50 | — | — | — |
| store | 12 | — | 30 | — | — | — |
| pull | — | — | 35 | — | — | — |
| take off | 71 | — | — | — | — | — |
| remove | 38 | — | — | — | — | — |
| collapse | 18 | — | — | — | — | — |
| unstack | 24 | — | — | — | — | — |

**Key observations:**
- **Object type strongly constrains verb choice.** Lights only get `turn on/off`; drawers get `open/close/pull`; switch gets `move`.
- Blocks are the only objects with diverse verb options (push/pick up/slide/rotate/grasp/lift/sweep/turn/stack/place/put).
- `push` and `slide` are the most polysemous — they apply to blocks, doors, drawers, AND switches.

---

## 7. Syntactic Templates (POS patterns)

Top templates over 259 unique instructions:

| Count | Template | Example |
|-------|----------|---------|
| 36 | VERB DET ADJ NOUN PREP DET NOUN | "push the blue block to the left" |
| 26 | VERB DET NOUN PREP DET NOUN | "place the block in the drawer" |
| 21 | VERB DET ADJ NOUN PREP DET ADJ | "slide the red block towards right" |
| 16 | VERB PREP DET ADJ NOUN | "turn on the green light" |
| 15 | VERB PREP DET ADJ NOUN PREP DET NOUN | "go towards the blue block in the drawer" |
| 13 | VERB DET ADJ NOUN VERB PREP DET NOUN | "grasp the blue block lying in the drawer" |
| 9 | VERB DET VERB NOUN PREP DET NOUN | "place the grasped object in the cabinet" |
| 6 | VERB DET NOUN | "close the drawer" |
| 6 | PREP DET NOUN VERB PREP DET ADJ NOUN | "in the cabinet pick up the red block" |

**Observations:**
- Most instructions follow **VERB [DET] [ADJ] NOUN [PP]** — standard imperative with optional modifiers
- Reduced relative clauses are common: "the block **lying** in the drawer" (spaCy parses `lying` as VERB/acl)
- Fronted PPs appear: "**in the cabinet** pick up the block" — unusual word order
- Compound instructions with `go...and VERB` use coordination

---

## 8. Example Full Dependency Parses

**"push the blue block to the left"**
```
push/VERB/ROOT → block/NOUN/dobj (the/DET/det, blue/ADJ/amod)
               → to/ADP/prep → left/ADJ/pobj (the/DET/det)
```

**"go towards the red block in the drawer and pick it up"**
```
go/VERB/ROOT → towards/ADP/prep → block/NOUN/pobj (the/DET, red/ADJ, in/ADP→drawer/NOUN)
             → pick up/VERB/conj (and/CCONJ/cc) → it/PRON/dobj
```

**"place the grasped object in the sliding cabinet"**
```
place/VERB/ROOT → object/NOUN/dobj (the/DET, grasped/VERB/amod)
                → in/ADP/prep → cabinet/NOUN/pobj (the/DET, sliding/VERB/amod)
```

**"take off the block that is on top of the other one"**
```
take off/VERB/ROOT → block/NOUN/dobj (the/DET)
                       → is/AUX/relcl (that/PRON/nsubj)
                           → on/ADP/prep → top/NOUN/pobj
                               → of/ADP/prep → one/NUM/pobj (the/DET, other/ADJ)
```

---

## 9. Summary: Compositional Structure of CALVIN Instructions

CALVIN instructions have a **shallow compositional structure** with 4 main axes of variation:

| Axis | Vocabulary | Cardinality |
|------|-----------|-------------|
| **Verb** | 22 classes (see `data/verb_classes.txt`); 4 are phrasal verbs (pick up, take off, turn on, turn off) | 22 |
| **Object** | block, door, drawer, switch, light/lamp/led, object (generic), blocks (stack) | ~7 types |
| **Object modifier** | colors (blue, red, pink, yellow, green), state (grasped, stacked, top), type (sliding, led) | 11 |
| **Spatial modifier** | left, right, up, down, into [location], on top, from [location], in [location] | ~10 patterns |

The **effective instruction space** is much smaller than the 259 surface forms suggest. Many instructions are paraphrases differing only in:
- Synonym verbs: {pick up, grasp, lift} all mean "acquire block"
- Location variants: {drawer, cabinet, slider, sliding cabinet} as synonymous containers
- Direction variants: {to the left, towards the left} are identical
- Color swaps: blue/red/pink are interchangeable for block actions

**Estimated distinct semantic actions:** ~34 unique (verb, object_type, direction) triples, further reducible to ~15 if verb synonyms are collapsed.

### Implications for Classification
1. **Object type is highly predictive of verb** — knowing it's a light immediately constrains to {turn on, turn off}
2. **Direction is orthogonal to verb** for most verbs — left/right doesn't change the action class
3. **Color is pure distractor** — no verb depends on block color. Non-color modifiers (`grasped`, `stacked`, `top`) weakly correlate with specific verbs (place/put/store, collapse/take off, remove) but these are low-frequency
4. **Location provides weak signal** — "from drawer" vs "from table" doesn't change the verb, but "into drawer" correlates with push/sweep/store
