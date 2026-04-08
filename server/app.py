from inference import app

def main():
    import uvicorn
    uvicorn.run("inference:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()