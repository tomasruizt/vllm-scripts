Symlink the AGENTS.md file to the project root to reuse it.

# Shell setup

Reusable Bash aliases and functions in .bashrc-d/

From this directory, copy the snippets into your shell configuration directory (`cp -i` asks before replacing existing files):

```bash
mkdir -p ~/.bashrc.d
cp -i .bashrc-d/* ~/.bashrc.d/
```

If `~/.bashrc` does not already load `~/.bashrc.d`, add:

```bash
for rc in "$HOME"/.bashrc.d/*; do
    if [ -f "$rc" ]; then
        . "$rc"
    fi
done
unset rc
```

Open a new Bash shell or run `source ~/.bashrc` to load the commands.

## Environment variables

- `HF_TOKEN`
- `HF_HUB_CACHE`

This directory is public. Keep tokens, credentials, signing keys, Git identity, and machine-specific paths in local configuration; do not copy entire dotfiles here.
