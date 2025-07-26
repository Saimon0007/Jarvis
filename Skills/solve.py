"""
Solve skill for Jarvis: Evaluates math expressions and simple Python code safely.
Usage: solve <expression>
"""
import math
import sympy
import os
import requests
from sympy import sympify, Symbol, Integral, Derivative, latex

WOLFRAM_APPID = os.getenv("WOLFRAMALPHA_APPID")

def wolframalpha_query(query):
    if not WOLFRAM_APPID:
        return None
    url = f"https://api.wolframalpha.com/v1/result?appid={WOLFRAM_APPID}&i={requests.utils.quote(query)}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        pass
    return None

def solve_skill(user_input):
    try:
        expr = user_input[len("solve"):].strip()
        expr = expr.replace("?", "").replace(",", "")
        expr = expr.replace("multiplied by", "*").replace("times", "*")
        expr = expr.replace("x", "*")
        expr = expr.replace("plus", "+").replace("add", "+")
        expr = expr.replace("minus", "-").replace("subtract", "-")
        expr = expr.replace("divided by", "/").replace("divide", "/")
        expr = expr.replace("power", "**").replace("to the power of", "**")
        expr = expr.strip()
        # Try SymPy for symbolic math
        try:
            # Handle integration and differentiation
            if "integrate" in expr or "∫" in expr:
                # Example: integrate x**2 dx
                parts = expr.replace("integrate", "").replace("∫", "").split("d")
                func = sympify(parts[0].strip())
                var = Symbol(parts[1].strip()) if len(parts) > 1 else Symbol("x")
                result = Integral(func, var).doit()
                return f"Integral: {latex(Integral(func, var))} = {result}"
            elif "differentiate" in expr or "derivative" in expr or "d/d" in expr:
                # Example: derivative x**2 wrt x
                if "wrt" in expr:
                    func, var = expr.split("wrt")
                    func = sympify(func.replace("derivative", "").replace("differentiate", "").replace("d/d", "").strip())
                    var = Symbol(var.strip())
                else:
                    func = sympify(expr.replace("derivative", "").replace("differentiate", "").replace("d/d", "").strip())
                    var = Symbol("x")
                result = Derivative(func, var).doit()
                return f"Derivative: {latex(Derivative(func, var))} = {result}"
            else:
                result = sympify(expr).evalf()
                return f"SymPy: {expr} = {result}"
        except Exception:
            pass
        # Try WolframAlpha if available and query looks complex
        if WOLFRAM_APPID and any(word in expr.lower() for word in ["integral", "derivative", "limit", "sum", "product", "solve", "root", "equation", "log", "sin", "cos", "tan", "arctan", "arcsin", "arccos", "diff", "integrate", "∫", "d/d"]):
            wa_result = wolframalpha_query(expr)
            if wa_result:
                return f"WolframAlpha: {wa_result}"
        # Only allow math and built-in safe functions
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"The answer is: {result}"
    except Exception as e:
        return f"Sorry, I couldn't solve that. Try a simpler math expression! ({e})"

def register(jarvis):
    jarvis.register_skill("solve", solve_skill)
