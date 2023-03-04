from pathlib import Path

import pytest
from awesome_list_miner.md_parser import find_all_links, is_github_repo_url, md_to_html

script_location = Path(__file__).absolute().parent


def read(file_name: str) -> str:
    f = open(script_location / "__mocks__/" / file_name, "r")
    return f.read()


simple_md = read("simple.md")


@pytest.mark.parametrize(
    "markdown,links",
    [
        (
            simple_md,
            [
                ("Table of Contents", "#table-of-contents"),
                ("Simple", "#simple"),
                ("folke/lazy.nvim", "https://github.com/folke/lazy.nvim"),
                (
                    "regular-expressions topic",
                    "https://github.com/topics/regular-expressions?l=python",
                ),
            ],
        ),
    ],
)
def test_find_all_links(markdown, links):
    assert find_all_links(md_to_html(markdown)) == links


@pytest.mark.parametrize(
    "url,check_result",
    [
        ("#simple", False),
        ("#table-of-contents", False),
        # It's a website but not a github repo.
        ("https://www.shorturl.at/shortener.php", False),
        ("https://github.com/folke/lazy.nvim", True),
        ("http://github.com/folke/lazy.nvim", True),  # With insecure protocol is fine.
        ("https://github.com/eleventigers/awesome-rxjava#readme", True),  # With anchor.
        # It's a github page but not a repo.
        ("https://github.com/topics/regular-expressions?l=python", False),
    ],
)
def test_is_github_repo_url(url: str, check_result: bool):
    assert is_github_repo_url(url) == check_result


@pytest.mark.parametrize(
    "markdown,links",
    [
        (
            simple_md,
            [
                ("folke/lazy.nvim", "https://github.com/folke/lazy.nvim"),
            ],
        ),
    ],
)
def test_find_all_github_repo_links(markdown, links):
    github_repo_links = filter(
        lambda link: is_github_repo_url(link[1]), find_all_links(md_to_html(markdown))
    )

    assert list(github_repo_links) == links
