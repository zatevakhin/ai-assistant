# AI Assistant

## Development using Nix [devenv](https://devenv.sh/)

This guide explains how to install Nix package manager and setup development environment using the Nix package manager.

## Installing Nix on Ubuntu or other distro that uses `systemd`.

> Same should work for MacOS but omit 3rd step.

1. Install Nix package manager:
```bash
sh <(curl -L https://nixos.org/nix/install) --daemon
```
> Yes, yes, yes, to the end. And that's basically done.

2. Enable Flakes support by creating or editing `~/.config/nix/nix.conf`:
```bash
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
```

3. Restart the Nix daemon:
```bash
sudo systemctl restart nix-daemon
```

### Install Nix [devenv](https://devenv.sh/)

Command below should be enough.
```bash
nix profile install nixpkgs#devenv
```

> If not, follow the official guide: https://devenv.sh/getting-started/

### Activate development environment
Run `nix develop` it will create environment like `conda`/`mamba`


