# AI usage

## Tools I used, and for which functions

I used ChatGPT to check the syntax and logic of my code and help me understand the task requirements. I wrote the `audit()` function myself and also wrote the precision-related function myself. I used VS Code to edit my code and pytest to test it.

## One place where the AI was wrong, incomplete, or not what the spec said

Some of the suggested code in `metrics.py` was incomplete and still contained `raise NotImplementedError`. I noticed this when I ran `uv run pytest -q` and saw test failures. I completed the missing code and ran the tests again.

## The function I wrote without AI, and why

I wrote `audit()` myself because I wanted to practise implementing the requirements independently. I used ChatGPT afterward to check its syntax and logic, which helped me catch mistakes while still writing the function myself.
