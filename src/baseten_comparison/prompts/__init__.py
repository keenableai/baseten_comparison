from jinja2 import Environment, PackageLoader

_env = Environment(
    loader=PackageLoader("baseten_comparison", "prompts"), keep_trailing_newline=False
)


def render(name: str, **context) -> str:
    return _env.get_template(f"{name}.jinja").render(**context).strip()
