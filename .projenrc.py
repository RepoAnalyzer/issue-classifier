import io
import subprocess
from typing import List, Tuple, Union

from projen import FileBase, IniFile, SampleDir, TomlFile, YamlFile
from projen.python import PythonProject

MODULE_NAME = "repolier"

# Folders holding all our modules with code.
ROOT_MODULE_DIR = "src"
ROOT_TEST_DIR = "tests"


def exec(command: List[str]) -> Tuple[bytes, bytes]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


class DosIniFile(FileBase):
    def __init__(self, *args, lines="", **options):
        super(DosIniFile, self).__init__(*args, **options)
        self.lines = lines

    def _synthesize_content(self, _) -> Union[str, None]:
        return self.lines


class PythonRepoAnalyzerProject(PythonProject):
    black = None
    flake8 = None
    isort = None

    pre_commit = None
    commitizen = None

    pytestConfig = None

    def __init__(
        self,
        black=True,
        flake8=True,
        isort=True,
        pre_commit=True,
        commitizen=True,
        **options,
    ):
        super(PythonRepoAnalyzerProject, self).__init__(**options)
        self.gitignore.exclude(".venv_win")

        if black:
            self.add_black()
        if flake8:
            self.add_flake8()
        if isort:
            self.add_isort()
        if pre_commit:
            self.add_pre_commit()
        if commitizen:
            self.add_commitizen()
        if self.pytest:
            self.add_pytest()

    def add_black(self):
        """Add black to the project as a dev dependency."""
        self.black = True
        self.add_dev_dependency("black@^22")

    def add_flake8(self):
        """Add flake8 to the project as a dev dependency.
        Create a simple configuration file to prevent conflicts with black.
        """
        self.flake8 = TomlFile(
            self,
            ".flake8",
            obj={
                "flake8": {
                    "max-line-length": 88,
                    "extend-ignore": "E203",
                },
            },
        )

        self.add_dev_dependency("flake8@^5")

    def add_isort(self):
        """Add isort to the project as a dev dependenncy.
        Create a simple configuration file to prevent conflicts with black.
        """
        self.isort = IniFile(
            self,
            ".isort.cfg",
            obj={
                "settings": {
                    "profile": "black",
                },
            },
        )

        self.add_dev_dependency("isort@^5.12.0")

    def add_pre_commit(self):
        """Add pre-commit the project as a dev dependenncy.
        Install github pre-commit hooks.
        Add base configuration for installed tools. List of supported modules:
            * Default:
                - trailing-whitespace,
                - end-of-file-fixer,
                - check-yaml,
                - check-added-large-files,

            * black,
            * flake8,
            * isort.
        """
        # Each string is in a regex format.
        pre_commit_targets = [
            rf"{ROOT_MODULE_DIR}\/.*",
            rf"{ROOT_TEST_DIR}\/.*",
            r"\.projenrc.py",
        ]
        contents = {
            # src and test files.
            "files": r"|".join(pre_commit_targets),
            "repos": [
                {
                    "repo": "https://github.com/pre-commit/pre-commit-hooks",
                    "rev": "v2.3.0",
                    "hooks": [
                        {"id": "trailing-whitespace"},
                        {"id": "end-of-file-fixer"},
                        {"id": "check-yaml"},
                        {"id": "check-added-large-files"},
                    ],
                }
            ],
        }

        if self.black:
            contents["repos"].append(
                {
                    "repo": "https://github.com/psf/black",
                    "rev": "22.10.0",
                    "hooks": [
                        {"id": "black"},
                    ],
                },
            )

        if self.flake8:
            contents["repos"].append(
                {
                    "repo": "https://github.com/pycqa/flake8",
                    "rev": "5.0.4",
                    "hooks": [
                        {"id": "flake8"},
                    ],
                },
            )

        if self.isort:
            contents["repos"].append(
                {
                    "repo": "https://github.com/PyCQA/isort",
                    "rev": "5.12.0",
                    "hooks": [
                        {"id": "isort"},
                    ],
                },
            )

        self.pre_commit = YamlFile(
            self,
            ".pre-commit-config.yaml",
            obj=contents,
        )

        self.add_dev_dependency("pre-commit@^2")

    def setup_pre_commit(self):
        command = ["pre-commit", "install"]
        self.logger.info("Setting up a git hook scropts for pre-commit...")
        self.logger.info(f"install | {' '.join(command)}")

        info, error = exec(command)

        if info:
            self.logger.info(info.decode("utf-8"))
        elif error:
            self.logger.error(error.decode("utf-8"))

    def add_commitizen(self):
        self.commitizen = True
        self.add_dev_dependency("commitizen@^2")

    def add_pytest(self):
        additional_options = [
            "--import-mode=importlib",
            "--cov",
            "--no-cov-on-fail",
            "--cov-branch",
            "--cov-report=term",
            "--cov-report=html",
            "--color=yes",
            "--code-highlight=yes",
            "--verbosity=2",
            "--no-header",
        ]
        from configparser import ConfigParser

        pytest_config = ConfigParser()
        obj = {
            "pytest": {
                "addopts": " ".join(additional_options),
                "testpaths": [ROOT_TEST_DIR],
                "log_cli": True,
            },
        }

        def convert_option_value_to_str(value):
            if type(value) != list:
                return str(value)

            return "\n" + "\n".join(value)

        for section, options in obj.items():
            pytest_config.add_section(section)
            for option, option_value in options.items():
                pytest_config.set(
                    section, option, convert_option_value_to_str(option_value)
                )

        with io.StringIO() as ss:
            pytest_config.write(ss)
            ss.seek(0)
            self.pytestConfig = DosIniFile(self, "pytest.ini", lines=ss.read())

    def post_synthesize(self):
        super(PythonRepoAnalyzerProject, self).post_synthesize()
        if self.pre_commit:
            self.setup_pre_commit()


project = PythonRepoAnalyzerProject(
    author_email="46250621+DeadlySquad13@users.noreply.github.com",
    author_name="DeadlySquad13",
    module_name=MODULE_NAME,
    name="projen_template",
    version="0.1.0",
    pytest_options={"testdir": f"{ROOT_TEST_DIR}/{MODULE_NAME}"},
    setuptools=True,
    sample=False,  # Overriden by custom sample dir.
)

# From root.
pythonLibSample = SampleDir(
    project,
    f"{ROOT_MODULE_DIR}/{project.module_name}",
    files={
        "__init__.py": '__version__ = "0.1.0"\n',
        "__main__.py": "\n".join(
            [
                f"from {project.module_name}.example import hello",
                "",
                'if __name__ == "__main__":',
                '    name = input("What is your name? ")',
                "    print(hello(name))",
                "",
            ]
        ),
        "example.py": "\n".join(
            [
                "def hello(name: str) -> str:",
                '    """A simple greeting.',
                "    Args:",
                "        name (str): Name to greet.",
                "    Returns:",
                "        str: greeting message",
                '    """',
                '    return f"Hello {name}!"',
                "",
            ]
        ),
    },
)

project.synth()
