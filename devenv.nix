{
  config,
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
      # OLLAMA_BASE_URL = "http://ollama.homeworld.lan";
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux {
      # FIX: NixOS User problems
      LD_LIBRARY_PATH = "/run/opengl-driver/lib:$LD_LIBRARY_PATH";
    }
    // {
      # NOTE: Just text colors.
      YELLOW = "\\033[0;33m";
      GREEN = "\\033[0;32m";
      RED = "\\033[0;31m";
      CLEAR = "\\033[0m";
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
    echo -e "Development environment is ready!\n"
    echo -e " - Run ${config.env.GREEN}docker compose up mumble -d${config.env.CLEAR} to spin-up Mumble server."
    echo -e " - Run ${config.env.GREEN}docker compose up ollama -d${config.env.CLEAR} to spin-up Ollama server (if needed)."
    echo -e " - Run ${config.env.GREEN}jupyter-ui${config.env.CLEAR} to spin-up Jupyter Notebook server."
    echo -e " - Finally run ${config.env.GREEN}python assistant/main.py${config.env.CLEAR} to run assistant."
    echo -e " - ${config.env.YELLOW}Don't forget${config.env.CLEAR} to cleanup running ${config.env.RED}docker containers${config.env.CLEAR} after you finish."
    echo -e "\n"
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

  scripts = {
    jupyter-ui.exec = "jupyter lab --no-browser --NotebookApp.token='' --NotebookApp.notebook_dir=$(pwd) $(pwd)/questions.ipynb";
  };

  # See full reference at https://devenv.sh/reference/options/
}
