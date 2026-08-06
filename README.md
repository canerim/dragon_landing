# DragonExam

Static site for DragonExam.

## Structure

```
.
├── index.html      # hero page
└── css/
    └── styles.css  # base styles + hero
```

## Running locally

No build step — open `index.html` directly, or serve the folder:

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

## Current state

Blockout only: bright purple page background and a centered red 16:9 box in a
100vh hero. No content, type, or components yet.

## Design tokens

Defined as CSS custom properties in `css/styles.css` under `:root`:

| Token          | Value     | Use                     |
| -------------- | --------- | ----------------------- |
| `--color-bg`   | `#8a2be2` | Page background         |
| `--color-box`  | `#ff2b2b` | Hero box                |
| `--font-body`  | Inter     | Body font (Google Fonts)|
| `--font-size-body` | `16px` | Body size              |
