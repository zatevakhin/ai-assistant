{ pkgs, lib, config, inputs, ... }:
{
  name = "ai-assistant";

  env.PYTHONPATH = ".";
  env.OLLAMA_BASE_URL = "http://galactica.lan:11434";
  # FIX: NixOS User problems
  env.LD_LIBRARY_PATH = "/run/opengl-driver/lib:$LD_LIBRARY_PATH";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.zsh
    pkgs.libopus
    pkgs.gcc-unwrapped.lib
    pkgs.cudaPackages.cudatoolkit
    pkgs.cudaPackages.libcublas
    pkgs.cudaPackages.cudnn
  ];

  # https://devenv.sh/scripts/
  # scripts.hello.exec = "";

  enterShell = '''';

  # https://devenv.sh/tests/
  enterTest = '''';

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
  processes.mumble.exec = "docker compose up mumble";

  # See full reference at https://devenv.sh/reference/options/
  starship.enable = false;
}
