---
name: LawnBerry Pi
description: A neon field-terminal for a real autonomous mower — cyan/magenta signal glow, chamfered panels, monospace readouts. Covers the app shell and Dashboard; task-oriented routes (Control/Settings/Maps/Planning/Mission Planner) run a second, calmer "Ready Room" dialect documented in prose below — see Overview.
colors:
  signal-cyan: "#00ffff"
  reactor-magenta: "#ff00ff"
  void-violet: "#2d1b69"
  void-black: "#0a0a0a"
  panel-navy: "#16213e"
  status-green: "#00ff00"
  status-yellow: "#ffff00"
  status-red: "#ff0040"
  status-orange: "#ff6600"
  focus-mint: "#00ff92"
typography:
  display:
    fontFamily: "'Orbitron', 'Courier New', monospace"
    fontSize: "2rem"
    fontWeight: 900
    lineHeight: 1.1
    letterSpacing: "3px"
  title:
    fontFamily: "'Courier New', 'Consolas', monospace"
    fontSize: "1.4rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "2px"
  label:
    fontFamily: "'Courier New', 'Consolas', monospace"
    fontSize: "0.9rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "2px"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
spacing:
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.panel-navy}"
    textColor: "{colors.signal-cyan}"
    rounded: "{rounded.md}"
    padding: "19px 24px"
  button-primary-hover:
    backgroundColor: "{colors.signal-cyan}"
    textColor: "#000000"
  card-primary:
    backgroundColor: "{colors.panel-navy}"
    textColor: "{colors.signal-cyan}"
    rounded: "{rounded.lg}"
---

# Design System: LawnBerry Pi

## Overview

**Creative North Star: "The Neon Field Terminal"**

The dashboard reads as a hardened diagnostic console bolted to real outdoor hardware, not a consumer app. Every surface is dark-void black shading into deep navy, lit from within by cyan signal glow with magenta as its rare second harmonic. Panels are chamfered like an equipment enclosure rather than softly rounded like a phone app; the interface is meant to feel monitored and alive — animated grid lines drift underneath, a scanline sweeps the header, metric values hum with an underline pulse — because the thing it's showing you is a machine that is actually moving outside, right now.

This is a single-operator instrument panel, not a marketed product surface (`PRODUCT.md`): legibility and honest machine-state read outrank charm. Nothing here should look calm when the mower isn't calm — the emergency-stop button's flashing red glow is not decoration, it's the same visual grammar as everything else turned up to its most urgent setting.

**Formerly confirmed anti-reference, now closed:** AI & Model Control and Telemetry were retheme onto the Neon Field Terminal (they anchor on `MetricWidget.vue`, already cyan); Tractor Control was retheme onto the Ready Room (it's the tractor-platform twin of `ControlView.vue`). No plain white admin-panel views remain — see the route map below.

**Two sanctioned dark systems, not one.** An earlier pass of this document overstated how far the Neon Field Terminal actually reaches. It is the truth for the persistent app shell (`App.vue`'s header/nav/footer/grid backdrop, present on every route), the Dashboard route's own card grid (`dashboard/*.vue`, `MetricWidget.vue`), and — as of the most recent pass — AI Control and Telemetry. Control, Settings, Maps, Planning, Mission Planner, and Tractor Control run a second, internally consistent dialect: **"The Ready Room"** (see below). Both are sanctioned; neither is drift against the other.

**Known residual seam, not yet resolved:** `components/RtkDiagnosticsPanel.vue` is a shared child component styled for the Ready Room (green accent, small radius) — correct for two of its three consumers (`SettingsView.vue`, `RtkDiagnosticsView.vue`, both Ready Room), but it's also embedded inside `TelemetryView.vue`, which is Neon Field Terminal (cyan). The panel renders correctly on its own terms but its green Refresh button and heading sit visibly against Telemetry's cyan card grid — a real, visible dialect seam confirmed by screenshot, not yet fixed. Resolving it cleanly needs either a per-parent color override (a `variant` prop) or accepting the seam as a deliberate "diagnostics panel is always green" exception; neither has been decided.

**Full route map:**
| Route | Dialect |
|---|---|
| Dashboard (`/`) | Neon Field Terminal |
| Login (`/login`) | Neon Field Terminal (retro-card shell; no nav/data chrome of its own) |
| AI Control (`/ai`) | Neon Field Terminal (anchors on `MetricWidget.vue`, 6 instances) |
| Telemetry (`/telemetry`) | Neon Field Terminal (anchors on `MetricWidget.vue`, 9 instances) — embeds the Ready Room-styled `RtkDiagnosticsPanel`, a known residual seam, see above |
| Control, Settings, Maps, Planning, Mission Planner, Tractor Control (`/tractor`) | Ready Room |
| RTK Diagnostics (`/rtk`) | Ready Room (`RtkDiagnosticsPanel` runs on `var(--secondary-dark)`/`var(--primary-light)`/`var(--accent-green)`, small radius, no chamfer — despite an unrelated `theme.css` backfill comment that might read otherwise) |
| Docs Hub (`/docs`) | Ready Room (its accent is `#00ff92`, `--accent-green`'s exact value; borders stay its own `#2c3e50` slate, not Signal Cyan — a Read-mode surface, not an instrument panel, so it doesn't take the Terminal's chrome-level cyan borders even though its accent matches the Ready Room's green) |

**Key Characteristics (Neon Field Terminal — chrome + Dashboard):**
- Void-black-to-navy surfaces lit by cyan glow, magenta used only as a rare second signal
- Chamfered (clipped-corner) panels and pills instead of soft rounded corners
- Heavy, continuous ambient motion: pulsing grid, scanning header sweep, breathing glows
- Monospace type on chrome and metric readouts — this is a readout, not a magazine
- Elevation is glow, not shadow

### The Ready Room (Control · Settings · Maps · Planning · Mission Planner)

Where the Neon Field Terminal is a readout you watch, the Ready Room is a console you work from — filling in schedules, drawing zones, adjusting settings, dispatching missions. It's the same machine, viewed from the desk instead of the field: still dark, still hardware-adjacent, but calmer and denser, because these screens ask the operator to read forms and make decisions rather than glance at a number.

**Key Characteristics:**
- Same void-black-to-navy neutrals as the Terminal, but **Go Mint** (`#00ff92`, `var(--accent-green)`) is the primary accent instead of cyan — success, active, and "you are here" states all key off it.
- Small border-radius everywhere (4–8px). No chamfer — clipped corners are chrome-only, reserved for the persistent nav/pills.
- Flat 1px borders and a hover color shift (border brightens to the accent) instead of glow shadows. No ambient motion, no scanline, no breathing glow.
- Body/UI typeface is the browser's default system sans-serif (this app's global `body` font — see Typography), not monospace. Denser, more legible for forms and tables at length; the Terminal's instrument-panel monospace stays where it earns its keep, on metrics and chrome.
- Status colors are tokenized in `assets/theme.css` (`--danger: #ff4343`, `--warning: #ffc107`, `--info: #17a2b8`, `--accent-green: #00ff92`) and shared across every Ready Room view. These are this dialect's own palette — don't substitute the Terminal's Emergency Red/Caution Yellow/Signal Cyan here, and don't pull the Ready Room's mint into a Dashboard card.

These tokens live in `theme.css`, not in this file's YAML frontmatter — the frontmatter and `.impeccable/design.json` sidecar describe the Neon Field Terminal only, so the automated drift detector keeps checking Dashboard/chrome work against cyan without false-flagging Ready Room work for not being cyan.

## Colors

Near-black voids lit by cyan signal glow, with a magenta second harmonic reserved for rare high-drama moments (title gradient, header scanline) and a deep violet used structurally in the header/footer gradient.

### Primary
- **Signal Cyan** (`#00ffff`): the dominant color of the entire system — borders, body text inside cards, icon glow, nav links, focus/hover glow. If a surface reads as "on", it's glowing cyan.

### Secondary
- **Reactor Magenta** (`#ff00ff`): the system's second harmonic. Appears only as a gradient partner to cyan (title text gradient, header top-edge scanline) — never as a standalone fill or border. Its rarity is the point.

### Tertiary
- **Void Violet** (`#2d1b69`): the structural midpoint of the header/footer gradient (`#1a1a1a → #2d1b69 → #0d0d0d`). Gives the chrome depth without introducing a third competing accent.

### Neutral
- **Void Black** (`#0a0a0a`): app background and the outer stops of every card gradient.
- **Panel Navy** (`#16213e`, with `#1a1a2e` / `#0f0f23` as gradient partners): the card surface itself — a three-stop diagonal gradient, never a flat fill.

### Status & Feedback
- **Go Green** (`#00ff00`): online/active indicators, the "start" control's hover state.
- **Focus Mint** (`#00ff92`): form-field focus rings and the boundary-editor's drawn geometry — a distinct green from Go Green, reserved for "you are actively editing this."
- **Caution Yellow** (`#ffff00`): warning indicators, the "pause" control's hover state, trend-neutral metric readouts.
- **Alert Orange** (`#ff6600`): the "stop" control's hover state — one step below emergency.
- **Emergency Red** (`#ff0040`): offline/error indicators and the emergency-stop control, which additionally flashes (`emergencyFlash`, 0.5s) rather than sitting static — the only color paired with a dedicated urgency animation.

### Named Rules
**The One Signal Rule.** Cyan is the interface's default voice. Every other color — magenta, violet, the status palette — earns its appearance by meaning something specific (a second harmonic, a structural gradient stop, a state). A screen that reaches for magenta or a status color without a reason has broken character.

## Typography

**Display Font:** `'Orbitron', 'Courier New', monospace`
**Body/Chrome Font:** `'Courier New', 'Consolas', monospace`

**Character:** Monospace everywhere, uppercase almost everywhere, wide letter-spacing — this is meant to read as instrumentation, not prose.

**Known gap:** `Orbitron` is declared on metric values and buttons but is never loaded (no `<link>`, no `@font-face` in `index.html`). Every browser without it installed locally silently falls through to `'Courier New', monospace` — so today's *actual* rendered typeface for both roles is the same monospace stack. Treat `Orbitron` as this system's committed intent, not yet its shipped reality; either load it (Google Fonts or self-hosted) or drop it from the stack, but don't let new work assume it's rendering.

**Scope:** this typography — including "monospace everywhere" — describes the app shell and Dashboard route only. The Ready Room dialect (Control, Settings, Maps, Planning, Mission Planner) intentionally uses the app's global `body` font (a system sans-serif stack, `main.css`) instead: denser and more legible for the forms, lists, and tables those views are built from. Don't add a monospace override to a Ready Room view to "match the system" — that would be importing a Dashboard convention into a dialect that deliberately doesn't use it.

### Hierarchy
- **Display** (900, 2rem, letter-spacing 3px, uppercase): metric readouts inside cards (`VELOCITY`, `42 MPH`) — the number the operator scans for first.
- **Title** (700, 1.4rem, letter-spacing 2px, uppercase, cyan→magenta→cyan gradient text): the app brand title only.
- **Card Header** (700, 1.1rem, letter-spacing 2px, uppercase, cyan with soft text-shadow glow): card/section titles.
- **Label** (600–700, 0.9rem, letter-spacing 0.7–2px, uppercase): nav links, buttons, status-bar text.
- **Body** (inherits card color, no dedicated prose role): this system has no paragraph-copy typography role — it displays data, not editorial text.

### Named Rules
**The All-Caps Instrument Rule.** Interactive and label text is uppercase with positive letter-spacing almost without exception. Sentence case reads as a foreign, softer register this system doesn't use.

## Layout

Single fixed app shell: a gradient header (nav + brand + theme toggle), a centered `max-width: 1400px` main content column, and a gradient footer status bar — all persistent across routes. `.app-main` uses `padding: 2rem` (`1rem` under 768px). A fixed `.retro-grid-bg` (animated cyan grid lines + soft radial magenta/cyan blooms) sits behind everything at `z-index: -1`, giving every route the same ambient "console" backdrop regardless of its own content — this part is universal.

What fills that shell differs by route. The Dashboard is a card grid of `.retro-card` instances (see Components), one per sensor/metric/control cluster. The Ready Room routes (Control, Settings, Maps, Planning, Mission Planner) are mostly forms, lists, and toolbars rather than card grids — structured for scanning and data entry, still sitting on the same void-black backdrop, just built from the Ready Room's own component set (see Ready Room Components, below) instead of `.retro-card`.

Responsive behavior collapses the nav into a hamburger under 768px and reduces the title/logo/nav-link sizing in two steps (`≤1200px`, `≤768px`) before the full mobile stack.

## Elevation & Depth

No traditional drop shadows for hierarchy. Depth is conveyed by **glow intensity**: cards sit on `box-shadow` stacks that combine an outer cyan glow, a tighter inner glow, and an inset highlight/shading — `backdrop-filter: blur(10px)` behind everything reinforces the sense of a lit panel over a dim void rather than a card over a page. Hover states don't just lighten, they intensify: `.retro-card:hover` lifts 4px and boosts every glow layer's spread and opacity together.

### Shadow Vocabulary
- **Card rest** (`0 8px 32px rgba(0,255,255,.3), 0 0 20px rgba(0,255,255,.2), inset 0 1px 0 rgba(255,255,255,.1), inset 0 0 30px rgba(0,255,255,.05)`): default card elevation.
- **Card hover** (`0 12px 40px rgba(0,255,255,.4), 0 0 30px rgba(0,255,255,.3), inset 0 1px 0 rgba(255,255,255,.2), inset 0 0 40px rgba(0,255,255,.1)`): interactive lift.
- **Emergency pulse** (`emergencyFlash`: alternates `0 0 20px rgba(255,0,64,.8)` ↔ `0 0 40px rgba(255,0,64,1)`, 0.5s infinite): the one shadow that is itself an animation, reserved for the emergency-stop control.

### Named Rules
**The Glow-Not-Shadow Rule.** Nothing in this system gets a neutral gray drop shadow. If an element needs to lift off the surface, it gets a colored glow keyed to its own state color (cyan by default, its status color otherwise). This rule is scoped to the Neon Field Terminal (chrome + Dashboard). The Ready Room dialect doesn't glow at all: its cards and buttons sit on flat 1px borders and communicate "elevated" or "active" with a border-color shift to `var(--accent-green)` on hover/focus, not a shadow of any kind. That's a real, deliberate difference — not an unfinished glow.

## Shapes

Two coexisting corner languages within the Neon Field Terminal, used deliberately for different things. **Cards and buttons** use conventional small-radius corners (4–8px) — see `rounded` tokens. **Nav links, the sign-in pill, and status-bar pills** are chamfered instead: `clip-path: polygon(6px 0, 100% 0, calc(100% - 6px) 100%, 0 100%)` (pills use a tighter 5px chamfer) — a beveled, cut-corner silhouette that reads as equipment rather than software chrome.

The Ready Room dialect doesn't use chamfer at all — every card, button, badge, and input there is small-radius (4–8px), full stop. Chamfer is chrome-only: it belongs to `App.vue`'s persistent nav/pills, which render on every route including Ready Room ones, but it doesn't spread into the Ready Room's own content.

### Named Rules
**The Chamfered Panel Rule.** Within the Neon Field Terminal: anything that behaves like a physical control or status readout (a nav tab, a status pill, a hardware-adjacent control) gets clipped corners, not rounded ones; anything that behaves like a content container (a card, a form field) gets a small border-radius instead. Don't mix the two on the same element. This rule doesn't reach into the Ready Room — see above.

## Components

### Buttons
- **Shape:** `border-radius: 6px` (`{rounded.md}`), never chamfered.
- **Primary (`.retro-btn`):** three-stop dark navy gradient background (`#1a1a2e → #16213e → #0f0f23`), 2px solid cyan border, cyan Orbitron-stack text, subtle rest glow (`0 4px 15px rgba(0,255,255,.2)`), a diagonal light-sweep pseudo-element that travels across on hover.
- **Hover/Focus:** background inverts to a cyan→black gradient, text goes near-black, glow intensifies to `0 0 20px rgba(0,255,255,.8)`.
- **State variants:** each mission-control action gets its own hover accent color while sharing the same shell — start (green), pause (yellow), stop (orange), emergency (red, plus the `emergencyFlash` animation instead of a static hover state).

### Cards (`.retro-card`)
- **Corner Style:** 8px radius (`{rounded.lg}`).
- **Background:** navy diagonal gradient (Panel Navy family), never flat.
- **Border:** 2px solid Signal Cyan.
- **Shadow Strategy:** see Elevation & Depth — glow stack, intensifies on hover with a 4px lift.
- **Header:** `rgba(0,255,255,.1)` wash, 1px cyan bottom border, uppercase Card Header type with text-shadow glow.
- **Signature detail:** a 1px top-edge gradient line that animates side to side (`borderScan`, 3s) — every card has a faint "still scanning" tell even at rest.

### Inputs / Fields
- **Style:** flat panel background (`var(--primary-dark)`), 1px border in the muted panel-light token, 4px radius — noticeably calmer than cards or buttons; fields are where the operator types, not where the system performs.
- **Focus:** border switches to Focus Mint with a soft `0 0 0 2px rgba(0,255,146,.2)` ring — deliberately a different green from the Go Green status color, so "you're editing" never reads as "system is online."

### Navigation
- **Style:** chamfered pill links (see Shapes), uppercase, 0.9rem, cyan on transparent-cyan wash; active/hover state adds a solid cyan border, glow, and text-shadow together — there's no separate "active" color, only "more glow."
- **Mobile:** collapses to a hamburger-triggered vertical stack at ≤768px; the chamfer and glow language stay intact, only the layout changes.

### Signature Component: the ambient grid backdrop
A fixed, full-viewport layer (`.retro-grid-bg`) behind every route: two 1px cyan grid-line gradients plus two soft radial magenta/cyan blooms, animated on two independent loops (a 6s opacity pulse, a 20s positional drift). It's the one element that never depends on route state — it's the room the interface lives in, not a piece of content.

## Ready Room Components

The second dialect (Control, Settings, Maps, Planning, Mission Planner). Tokens live in `assets/theme.css`, not in this file's frontmatter.

- **Buttons (`.btn`, `.btn-primary`, `.btn-success`, `.btn-warning`, `.btn-danger`, `.btn-info`):** flat fill, `border-radius: 6px`, no border, no glow. `.btn-primary` uses `var(--accent-green)` background with `var(--primary-dark)` text (a bright-on-dark pairing, not white-on-color — plain white text on the mint accent fails contrast). `.btn-success` matches `.btn-primary`'s pairing. `.btn-warning` is `var(--warning)` with black text. `.btn-danger`/`.btn-info` use `var(--danger)`/`var(--info)` with white text. Hover is a `translateY(-2px)` lift, not a color invert.
- **Cards (`.card`, `.job-item`, `.zone-card`, etc.):** `var(--primary-dark)` background, `1px solid var(--primary-light)` border, 8px radius. Hover/active swaps the border to `var(--accent-green)` — no shadow, no lift beyond an occasional `translateY(-2px)` on directly-clickable cards.
- **Status badges/pills (`.status-*`, `.priority-*`, `.condition-*`, `.level-*`):** a translucent tint of the status color as background (`rgba(r,g,b,0.2)`) with a matching solid border and text color, keyed to `var(--accent-green)`/`var(--danger)`/`var(--warning)`/`var(--info)`. Small radius (4px), never chamfered.
- **Inputs:** same flat-panel/small-radius language as the Neon Field Terminal's Field Input, but in the system sans-serif rather than monospace, and focus rings key to `var(--accent-green)` rather than Focus Mint (the Ready Room doesn't distinguish a separate "editing" green from its "success" green the way the Terminal does).

## Do's and Don'ts

### Do:
- **Do** pick the dialect by route, not by habit: a new Dashboard panel or shell element defaults to the Neon Field Terminal (void-black/navy gradient surface, 2px cyan border, glow-based elevation, monospace chrome type); a new Control/Settings/Maps/Planning/Mission-Planner panel defaults to the Ready Room (`theme.css` tokens, small radius, flat borders, system sans-serif).
- **Do** reserve magenta for a genuine second-harmonic moment (a gradient partner to cyan), never as a standalone color — this applies within the Neon Field Terminal; the Ready Room doesn't use magenta at all.
- **Do**, within the Neon Field Terminal, give physical/hardware-adjacent controls (nav tabs, status pills) chamfered corners and content containers (cards, inputs) small border-radius instead — never mix the two on one element. The Ready Room skips chamfer entirely; don't introduce it there.
- **Do** key any Neon Field Terminal glow/shadow to the element's own state color, not a neutral gray; key any Ready Room border-hover state to `var(--accent-green)`, `var(--danger)`, `var(--warning)`, or `var(--info)` from `theme.css`, not a new one-off hex.
- **Do** treat the emergency-stop control as visually distinct from every other button: it's the only one allowed a continuous alarm animation.

### Don't:
- **Don't** build a new view in the plain white Bootstrap-admin style (`var(--card-bg,#fff)`, `0 1px 3px` shadow, pill badges) — that pocket existed in AI Control, Tractor Control, and Telemetry until this pass retheme all three onto the two sanctioned dialects; there's no longer a live example of it anywhere in the app, and it was never a third sanctioned language.
- **Don't** assume `'Orbitron'` is actually rendering — it isn't loaded anywhere in `index.html`; design against the monospace fallback unless you also add the font.
- **Don't** give a Neon Field Terminal card, button, or panel a plain neutral gray drop shadow — every elevated surface there glows in its own color instead. (The Ready Room has no glow rule to violate — its elevation language is flat borders by design, not an unfinished version of glow.)
- **Don't** add a static, non-animated focal element to a Neon Field Terminal screen that's otherwise full of ambient motion (grid pulse, scanline, breathing glows) — it will read as broken rather than calm. The Ready Room has no ambient motion to match in the first place.
- **Don't** hardcode a new status-color hex (a red, green, yellow, or blue) in a Ready Room file — `--danger`/`--warning`/`--info`/`--accent-green` already exist in `theme.css` for exactly this, and a fresh literal recreates the duplication this document was just corrected to stop.
