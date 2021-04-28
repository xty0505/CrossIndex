from pymapd import connect


class OmnisciManager(object):
    def __init__(self, properties=None):
        if properties is None:
            properties['user'] = 'admin'
            properties['password'] = 'HyperInteractive'
            properties['host'] = 'localhost'
            properties['dbname'] = 'crossindex'
        self.user = properties['user']
        self.password = properties['password']
        self.host = properties['host']
        self.dbname = properties['dbname']
        self.con = connect(user=self.user, password=self.password, host=self.host, dbname=self.dbname)

    def get_df(self, sql):
        return self.con.select_ipc(sql)

    def close(self):
        self.con.close()
