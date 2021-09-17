class NedSyntaxError(Exception):
    def __init__(self, line_num, col_num, err_msg):
        super(NedSyntaxError, self).__init__("line %d - column %d\n%s" % (line_num, col_num, err_msg))

        self.line_num = line_num
        self.col_num = col_num
