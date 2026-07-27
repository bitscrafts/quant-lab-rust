# Book Style Guide — Consistency Checklist

This document defines the required structure and style for each chapter in the
quant-finance book. Use it when adding new chapters or reviewing existing ones.

## Chapter Structure Template

Every chapter MUST follow this structure:

```latex
\chapter{<Title>}
\label{ch:<label>}

\begin{companioncode}{<crate-name>}
Code: \texttt{crates/<crate>/}\\
Dependencies: \crate{<dep1>}, \crate{<dep2>}\\
Run: \texttt{cargo run -p <crate> --example <example>}
\end{companioncode}

\section{Learning Objectives}
...
\section{<Content Sections>}
...
\section{Exercises}
...
\section{Key Takeaways}
\begin{itemize}
    \item \textbf{Concept}: explanation
\end{itemize}

\vspace{1em}
\noindent\textbf{Next chapter}: <preview of next chapter>
```

## Required Elements Checklist

For each chapter, verify:

### Header
- [ ] `\chapter{Title}` declaration
- [ ] `\label{ch:<label>}` for cross-references
- [ ] `\begin{companioncode}{crate}` with Code/Dependencies/Run lines

### Sections
- [ ] `\section{Learning Objectives}` with bullet points
- [ ] Content sections with inline Rust code
- [ ] `\section{Exercises}` with numbered problems
- [ ] `\section{Key Takeaways}` (not "Summary")

### Footer
- [ ] `\vspace{1em}\noindent\textbf{Next chapter}:` preview

## Code Block Guidelines

### Rust Implementation Code

Every major concept MUST have an inline Rust implementation:

```latex
\begin{lstlisting}[language=Rust]
pub fn function_name(...) -> Result<..., Error> {
    // implementation
}
\end{lstlisting}
```

**Required code blocks per concept:**
| Concept | Code Shows |
|---------|-----------|
| Data structure | `struct` definition |
| Algorithm | Core function implementation |
| Trait | Trait definition + impl block |
| Validation | Error handling pattern |

### CLI Example Output

```latex
\begin{lstlisting}[language=bash]
$ cargo run -p <crate> --example <name>
<output>
\end{lstlisting}
```

## Margin Notes

Use margin notes for formulas and quick references:

```latex
\marginnote{%
\textbf{Formula Name}\\[0.3em]
$x = \dfrac{a}{b}$\\[0.5em]
where $a$ is ... and $b$ is ...
}
```

## Key Insight Blocks

Highlight important takeaways:

```latex
\begin{keyinsight}
<Important concept that ties together the section>
\end{keyinsight}
```

## Warning Blocks

Flag common mistakes or caveats:

```latex
\begin{warning}
<Important caveat or common mistake>
\end{warning}
```

## Analysis Process for Book Maintenance

When reviewing a chapter for consistency:

### Step 1: Check Structure
```bash
# Verify chapter header exists
grep -n "\\\\chapter{" book/chapters/ch<N>.tex

# Verify companion code block
grep -n "companioncode" book/chapters/ch<N>.tex

# Verify Key Takeaways section
grep -n "Key Takeaways" book/chapters/ch<N>.tex

# Verify Next chapter link
grep -n "Next chapter" book/chapters/ch<N>.tex
```

### Step 2: Count Code Blocks
```bash
# Count Rust code blocks
grep -c "begin{lstlisting}" book/chapters/ch<N>.tex

# Compare with adjacent chapters (should be similar)
for f in book/chapters/ch*.tex; do
  echo "$f: $(grep -c 'begin{lstlisting}' $f) code blocks"
done
```

### Step 3: Verify Code Coverage

For each major concept in the chapter:
1. Read the corresponding source file in `crates/<crate>/src/`
2. Verify the chapter has an inline `\begin{lstlisting}[language=Rust]` showing the implementation
3. If missing, add a simplified version of the key function

### Step 4: Cross-Reference Check
```bash
# Find all \cref and \ref usage
grep -n "\\\\cref\|\\\\ref" book/chapters/ch<N>.tex

# Verify targets exist
grep -n "\\\\label{" book/chapters/ch*.tex
```

## Style Consistency Rules

1. **Section titles**: Use title case ("Learning Objectives", not "learning objectives")
2. **Code blocks**: Always specify `[language=Rust]` or `[language=bash]`
3. **Formulas**: Use `\dfrac` in margin notes, regular `\frac` in body
4. **Key Takeaways**: Use `\textbf{Concept}:` format for each bullet
5. **Dashes**: Use `---` for em-dash, `--` for en-dash

## Current Chapter Status

| Chapter | Header | Companion | Code Blocks | Takeaways | Next Link |
|---------|--------|-----------|-------------|-----------|-----------|
| ch01 | ✓ | ✓ | 9 | ✓ | ✓ |
| ch02 | ✓ | ✓ | 10 | ✓ | ✓ |
| ch03 | ✓ | ✓ | 8 | ✓ | ✓ |
| ch04 | ✓ | ✓ | 8 | ✓ | ✓ |
| ch05 | ✓ | ✓ | 9 | ✓ | ✓ |
| ch06 | ✓ | ✓ | 10 | ✓ | ✓ |
| ch07 | ✓ | ✓ | 7 | ✓ | ✓ |
| ch08 | ✓ | ✓ | 8 | ✓ | ✓ |
