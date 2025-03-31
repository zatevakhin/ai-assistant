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
  name = "system-iii";

  env =
    {
      PYTHONPATH = ".";
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux {
      # FIX: NixOS User problems
      LD_LIBRARY_PATH = "/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath cuda}:$LD_LIBRARY_PATH";
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
      curl
      libz
      libopus
      ffmpeg
      gcc-unwrapped.lib
    ]
    ++ cuda;

  enterShell = ''
    echo -e "Development environment is ready!\n"
    if docker ps | grep -q "mumblevoip/mumble-server:latest"; then
        echo -e " - ✅ Docker container with$GREEN Mumble server$CLEAR already running in background."
        MUMBLE_RUNNING=true
    else
        echo -e " - Run$GREEN docker compose up mumble -d$CLEAR to spin-up Mumble server."
        MUMBLE_RUNNING=false
    fi

    OLLAMA_RUNNING=false
    if [[ "$OLLAMA_BASE_URL" == *"localhost"* ]] || [[ "$OLLAMA_BASE_URL" == *"127.0.0.1"* ]]; then
        if docker ps | grep -q "ollama/ollama" && curl -s --head --fail "$OLLAMA_BASE_URL" > /dev/null; then
            echo -e " - ✅$GREEN Ollama server$CLEAR at $GREEN$OLLAMA_BASE_URL$CLEAR is available."
            OLLAMA_RUNNING=true
        elif docker ps | grep -q "ollama/ollama"; then
            echo -e " - ✅ Docker container with$GREEN Ollama server$CLEAR running, but service not responding."
            echo -e "   - Try running$GREEN docker restart ollama$CLEAR to fix it."
            OLLAMA_RUNNING=true
        else
            echo -e " - Run$GREEN docker compose up ollama -d$CLEAR to spin-up Ollama server."
        fi
    else
        if curl -s --head --fail "$OLLAMA_BASE_URL" > /dev/null; then
            echo -e " - ✅$GREEN Remote Ollama server$CLEAR at $GREEN$OLLAMA_BASE_URL$CLEAR is available."
        else
            echo -e " - ❌$RED Remote Ollama server$CLEAR at $RED$OLLAMA_BASE_URL$CLEAR is not responding."
            echo -e "   - Check if the remote server is running or if there are network/firewall issues."
            echo -e "   - Alternatively, use a local server with by un-commenting$GREEN OLLAMA_BASE_URL=http://localhost:11434$CLEAR in$GREEN devenv.nix$CLEAR"
        fi
    fi

    echo -e " - Run$GREEN jupyter-ui$CLEAR to spin-up Jupyter Notebook server for tests."
    echo -e " - Finally run$GREEN python assistant/main.py$CLEAR to run assistant."
    if $MUMBLE_RUNNING && $OLLAMA_RUNNING; then
        echo -e " -$YELLOW Don't forget$CLEAR to cleanup running$RED Mumble and Ollama containers$CLEAR after you finish."
        echo -e "   - Use$GREEN docker stop $(docker ps -q --filter ancestor=mumblevoip/mumble-server:latest) $(docker ps -q --filter ancestor=ollama/ollama)$CLEAR"
    elif $MUMBLE_RUNNING; then
        echo -e " -$YELLOW Don't forget$CLEAR to cleanup running$RED Mumble container$CLEAR after you finish."
        echo -e "   - Use$GREEN docker stop $(docker ps -q --filter ancestor=mumblevoip/mumble-server:latest)$CLEAR"
    elif $OLLAMA_RUNNING; then
        echo -e " -$YELLOW Don't forget$CLEAR to cleanup running$RED Ollama container$CLEAR after you finish."
        echo -e "   - Use$GREEN docker stop $(docker ps -q --filter ancestor=ollama/ollama)$CLEAR"
    else
        echo -e " - No cleanup needed as no containers are currently running."
    fi
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
