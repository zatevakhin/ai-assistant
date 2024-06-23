{ pkgs, lib, config, inputs, ... }:

{
  name = "ai-assistant";

  # https://devenv.sh/packages/
  packages = [ pkgs.git pkgs.zsh pkgs.libopus ];

  # https://devenv.sh/scripts/
  # scripts.hello.exec = "echo hello from $GREET";

  enterShell = ''

  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep "2.42.0"
  '';

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.11";

    venv = {
      enable = true;
      quiet = true;
      requirements = builtins.readFile ./requirements.txt;
    };
  };
  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
  starship.enable = true;
}
