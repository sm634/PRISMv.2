import pandas as pd
import chardet
import yaml
import os
import json


class FileHandler:
    def __init__(self):

        self.config = None
        # attributes for prompt files.
        self.prompt = ''
        self.prompts_folder_path = 'prompts/prompt_templates/'

        # attributes for queries files.
        self.queries_folder_path = 'queries/'
        self.queries_json = {}

        # attributes for data files.
        self.data = pd.DataFrame()
        self.data_input_folder_path = 'data/input/'
        self.data_output_folder_path = 'data/output/'
        self.model_config_file_path = 'configs/models_config.yaml'
        self.elasticsearch_config_file_path = 'configs/elasticsearch_config.yaml'

    @staticmethod
    def __get_file_encoding(file_path):
        """
        Function to identify the encoding of the file so that it can be used to read the file for further processing.
        :param file_path: name of the file to be read. This by default should sit in the data/input folder.
        :return: the encoding scheme of the file
        """

        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            return result['encoding']

    def get_prompt_from_file(self, file_name):
        """
        Retrieves the content from a prompt template .txt file and stores it in a variable as type string.
        :param file_name: the name of the file with the prompt template.
        'prompt_templates/red_flags_prompts1.txt'
        :return: the prompt template as string.
        """
        # ensure we read a .txt file to get the prompt template.
        assert '.txt' in file_name, ("The file is not of extension .txt. Please ensure it is, or include the extension"
                                     "in the argument to pass to this function.")

        file_path = self.prompts_folder_path + file_name
        # read the file.
        with open(file_path, 'r') as f:
            prompt = f.read()

        self.prompt = prompt

    def get_queries_from_json(self, file_name):
        """
        Retrieves the json file containing the collection metadata and queries from a .json file.
        :param file_name: the name of the file with the prompt template.
        'prompt_templates/red_flags_prompts1.txt'
        :return: the prompt template as string.
        """
        # ensure that the file is of type json.
        assert '.json' in file_name, ("The file is not of extension .json. Please ensure it is, or include the "
                                      "extension in the argument to pass to this function.")

        file_path = self.queries_folder_path + file_name
        # read the file and load it as dic.
        with open(file_path, 'r') as f:
            queries_json = f.read()
            queries_json = json.loads(queries_json)

        if isinstance(queries_json, str):
            queries_json = json.loads(queries_json)

        self.queries_json = queries_json

    def get_data_from_file(self, file_name):
        """
        Function to retrieve data as a pandas DataFrame from the designated data input folder.
        :param file_name: the name of the file (.csv) only currently.
        :return: a pandas DataFrame of the tabular data.
        """
        assert '.csv' in file_name, ("The file is not of extension .csv. Please ensure it is, or include the extension"
                                     "in the argument.")

        # set file path
        file_path = self.data_input_folder_path + file_name
        # get the file encoding so it can read correctly.
        file_encoding = self.__get_file_encoding(file_path)

        # read the file
        df = pd.read_csv(file_path, encoding=file_encoding)
        self.data = df

    def save_df_to_file(self, df, file_name):
        """
        Function to retrieve data as a pandas DataFrame from the designated data input folder. The standard encoding
        scheme that the file will be saved in is UTF-8.
        :param df: the pandas DataFrame to save as a csv.
        :param file_name: the name of the csv file
        :return: a pandas DataFrame of the tabular data.
        """

        file_path = self.data_output_folder_path + file_name
        df.to_csv(file_path, encoding='UTF-8', index=False)

    def get_prompt_template(self, file_name):
        """
        Get prompt template from file.
        :param file_name: the name of the file for the prompt template, including the .txt extension.
        :return: prompt str
        """
        self.get_prompt_from_file(file_name)
        prompt_template = self.prompt
        return prompt_template

    def get_df_from_file(self, file_name):
        """
        Get data dataframe from file.
        :param file_name: the name of the file for the data to move to DataFrame.
        :return: pandas DataFrame
        """
        self.get_data_from_file(file_name=file_name)
        df = self.data
        return df

    def get_config(self, config_type='model'):
        # get the correct file.
        config = ''
        if config_type == 'model':
            config = self.model_config_file_path
        elif config_type == 'elasticsearch':
            config = self.elasticsearch_config_file_path

        try:
            # get the configs with relative to example_main.py script.
            with open(config, 'r') as file:
                self.config = yaml.safe_load(file)

        except FileNotFoundError:
            # get file based on the path provided from .env file.
            config_path = os.environ['CONFIG_PATH']
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)

    def write_config(self, config_type='model'):
        # get the correct file.
        config = ''
        if config_type == 'model':
            config = self.model_config_file_path
        elif config_type == 'elasticsearch':
            config = self.elasticsearch_config_file_path

        try:
            # write the configs to the relative path file
            with open(config, 'w') as file:
                yaml.dump(config, file)

        except FileNotFoundError:
            # write file based on the path provided from .env file.
            config_path = os.environ['CONFIG_PATH']
            with open(config_path, 'w') as file:
                yaml.dump(config, file)
