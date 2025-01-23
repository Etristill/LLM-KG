from setuptools import setup, find_packages

setup(
    name="cognitive_mcts",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'networkx',
        'anthropic',
        'openai',
        'python-dotenv',
        'sweetbean'
    ],
)