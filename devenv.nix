{
  pkgs,
  lib,
  ...
}: let
  cuda =
    if pkgs.stdenv.isLinux
    then
      with pkgs.cudaPackages; [
        cudatoolkit
        libcublas
        cudnn
      ]
    else [];
in {
  name = "ai-assistant";

  env =
    {
      PYTHONPATH = ".";
      OLLAMA_BASE_URL = "http://localhost:11434";
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux {
      # FIX: NixOS User problems
      LD_LIBRARY_PATH = "/run/opengl-driver/lib:$LD_LIBRARY_PATH";
    };

  # https://devenv.sh/packages/
  packages = with pkgs;
    [
      git
      zsh
      libopus
      gcc-unwrapped.lib
    ]
    ++ cuda;

  enterShell = ''
    echo -e "Ready for development.\n"
  '';

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.11";

    poetry = {
      enable = true;
      activate.enable = true;
      install = {
        enable = true;
        allExtras = true;
      };
    };
  };

  # https://devenv.sh/processes/
  processes = {
    mumble.exec = "docker compose up mumble";
    ollama.exec = "docker compose up ollama";
    # TODO: Add jupyter lab web ui. (why? because i use neovim and can't use *.ipynb)
  };

  # See full reference at https://devenv.sh/reference/options/
}
