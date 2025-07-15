from pipeline import LinkPredictionPipeline
from pipeline_full_graph import LinkPredictionFullGraphPipeline

if __name__ == "__main__":
    user_input = input("Do you want to run the full graph pipeline? (y/n): ")
    if user_input.lower() == 'y':
        LinkPredictionFullGraphPipeline().run()
    else:
        LinkPredictionPipeline().run()
