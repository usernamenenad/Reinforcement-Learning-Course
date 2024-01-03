# Self-learning and adaptive algorithms (*Reinforcement learning*)

A list of my implementations of homework tasks for '***Self-learning and adaptive algorithms***' course.

## Running a certain homework task

Clone the repository using command 

```bash
git clone https://github.com/usernamenenad/Reinforcement-Learning-Course.git
```

and position yourself in a homework folder of interest.

Then, it is necessary to make a `python` virtual environment by executing 

```bash
python -m venv .venv
```

and activate it based on OS you're current on. If that's `Windows`, execute in `PowerShell`

```bash
.\.venv\Scripts\activate.ps1
```

If you're on `Linux`, activate the virtual environment by executing in `bash`

```bash
source ./.venv/bin/activate
```

A list of dependencies is located in `requirements.txt` and it is necessary to install required packages by executing 

```bash
pip install -r requirements.txt
```

If you're using *PyCharm* IDE, it will automatically install required packages for you.

Homeworks are constructed in such a way as to run them as ***test files***. After setting up your virtual environment, *pytest* is automatically installed, so you position yourself in a desired homework and run

```bash
pytest -s
```

This will run all tests. If you want to run a specific test, execute

```bash
pytest -s ./tests/`TESTNAME`
```
