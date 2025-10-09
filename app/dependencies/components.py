
from app.bootstrap.components import Components


def get_components(
        env: str = 'development',
        config_path: str = 'config.yaml'
) -> Components:
    return Components(env, config_path)