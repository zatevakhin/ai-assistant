{ pkgs, lib, config, inputs, ... }:
{
  name = "ai-assistant";

  env.PYTHONPATH = ".";
  env.OLLAMA_BASE_URL = "http://galactica.lan:11434";

  # https://devenv.sh/packages/
  packages = [ pkgs.git pkgs.zsh pkgs.libopus ];

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
