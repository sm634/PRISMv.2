from io import StringIO
import pandas as pd
import ibm_boto3
from ibm_botocore.client import Config


class IbmCos:
    def __init__(self,
                 cos_api_key,
                 cos_instance_crn,
                 auth_endpoint,
                 cos_endpoint,
                 bucket_name,
                 object):
        """
        Instantiate the IBM cloud object storage connector class to connect to COS objects
        :param cos_api_key: valid api key to connect to cos
        :param cos_instance_crn: cos instance crn
        :param auth_endpoint: authentication endpoint
        :param cos_endpoint: cos endpoint
        :param bucket_name: bucket name
        :param object: object name in the bucket
        """

        self.COS_API_KEY = cos_api_key
        self.COS_INSTANCE_CRN = cos_instance_crn
        self.AUTH_ENDPOINT = auth_endpoint
        self.COS_ENDPOINT = cos_endpoint
        self.BUCKET_NAME = bucket_name
        self.OBJECT = object

    def __connect_to_cos(self):

        # Create COS client
        cos_client = ibm_boto3.client(service_name="s3",
                                      ibm_api_key_id=self.COS_API_KEY,
                                      ibm_service_instance_id=self.COS_INSTANCE_CRN,
                                      ibm_auth_endpoint=self.AUTH_ENDPOINT,
                                      config=Config(signature_version='oauth'),
                                      endpoint_url=self.COS_ENDPOINT
                                      )
        return cos_client

    def get_object_df(self):
        """
        A function to fetch the cloud object and return a pandas DataFrame of it.
        :return: pandas DataFrame
        """

        # connect to cloud object storage using credentials.
        cos_client = self.__connect_to_cos()

        file_object = cos_client.get_object(
            Bucket=self.BUCKET_NAME,
            Key=self.OBJECT)

        try:
            data = file_object['Body'].read().decode('latin-1')  # Read the content of the object and decode it as UTF-8
        except UnicodeDecodeError:
            data = file_object['Body'].read().decode('latin-1')  # Read the content of the object and decode it as
            # latin-1

        df = pd.read_csv(StringIO(data))

        return df


