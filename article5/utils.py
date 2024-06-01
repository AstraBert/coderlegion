class NeuralSearcher:
    def __init__(self, collection, encoder):
        self.collection = collection
        self.encoder = encoder
    def search(self, text):
        results = self.collection.query(
            data=self.encoder.encode(text).tolist(),  # required
            limit=1,                     # number of records to return
            filters={},                  # metadata filters
            measure="cosine_distance",   # distance measure to use
            include_value=True,         # should distance measure values be returned?
            include_metadata=True,      # should record metadata be returned?
        )
        return results