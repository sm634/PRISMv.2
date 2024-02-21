from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from dotenv import load_dotenv
import os
from typing import List
from utils.files_handler import FileHandler


class WatsonDiscoveryV2Connector:

    def __init__(self):
        """
        Watson Discovery V2 connector class.
        """
        # we will get the connection keys from the .env file by loading it to the environment variables.
        load_dotenv()
        self.authenticator = IAMAuthenticator(os.environ['WATSON_DISCOVERY_APIKEY'])
        self.discovery_instance = DiscoveryV2(
            version=os.environ['WATSON_DISCOVERY_VERSION'],
            authenticator=self.authenticator
        )
        self.discovery_instance.set_service_url(os.environ['WATSON_DISCOVERY_URL'])
        self.project_id = os.environ['WATSON_DISCOVERY_PROJECT_ID']

        # get passage details from config
        file_handler = FileHandler()
        file_handler.get_config(config_file_name='elasticsearch_config')
        self.config = file_handler.config

        # get parameter values after parsing config.
        self.passages_config = self.config['WATSON_DISCOVERY_V2']['passages']
        self.max_per_document = self.passages_config['max_per_document']
        self.characters = self.passages_config['characters']

        self.response = None  # to be reassigned with response json packet.

    def query_response(self, query, collection_ids: List[str]):
        """Query results from Discovery collections.
        :param query: The query used for search.
        :param collection_ids: the set of collections to send the query request to."""

        response = self.discovery_instance.query(
            project_id=self.project_id,
            collection_ids=collection_ids,
            passages=self.passages_config,
            natural_language_query=query
        ).get_result()

        self.response = response

    def __get_results(self):
        """
        A hidden method that fetches the results of running the query.
        :return: the result json.
        """
        assert self.response is not None, "Please run the 'get_query_response' method before using this method."
        return self.response['results']

    def __get_kv_from_result(self, key: str):
        """
        Returns a list of all the items form a metadata. For instance the returned 'result' from the response,
        which is generated using the __get_results(self) method may or may not have certain key-value pairs.
        For instance, subtitle or table may appear as a key that can be accessed to retrieve the associated value for
        certain retrieved response results or not.
        :param key: Str the key to be searched for in the results output.
        :return: List of all values associated with the key (metadata)
        """
        # get all results
        results = self.__get_results()
        # store output.
        output = []

        for i in range(0, len(results)):
            try:
                output.append(results[i][key])
            except KeyError:
                # exception needed as not all desired keys will be retrieved for all retrieved data.
                output.append(f"THERE IS NO {key} FOR THIS DATA")

        return output

    def get_document_ids(self):
        """
        A function that grabs the document ids from the results of the query response.
        :return: List[document ids]
        """
        document_ids = self.__get_kv_from_result(key='document_id')
        return document_ids

    def get_result_metadata(self):
        """
        A function that grabs the result metadata from the results of the query response.
        :return: List[result metadata]
        """
        result_metadata = self.__get_kv_from_result(key='result_metadata')
        return result_metadata

    def get_subtitle(self):
        """
        A function that grabs the subtitles from the results of the query response.
        :return: List[subtitles]
        """
        subtitles = self.__get_kv_from_result(key='subtitle')
        return subtitles

    def get_document_passages(self):
        """
        A function that grabs the document passages from the results of the query response.
        :return: List[passages]
        """
        passages = self.__get_kv_from_result(key='document_passages')
        return passages

    def get_text(self):
        """
        A function that grabs the text from the results of the query response.
        :return: List[text]
        """
        text = self.__get_kv_from_result(key='text')
        return text

    def get_table(self):
        """
        A function that grabs the table data from the results of the query response.
        :return: List[table]
        """
        table = self.__get_kv_from_result(key='table')
        return table
