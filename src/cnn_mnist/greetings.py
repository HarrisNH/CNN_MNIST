import typer
app = typer.Typer()

@app.command()
def greetings(count: int = 1):
    for i in range(count):
        print("Hello World")
if __name__ == "__main__":
    app()
