#!/usr/bin/env python
#
#    loadcsv.py Load a file with CSV data into a database
#    Copyright (C) 2009  Ferran Pegueroles <ferran@pegueroles.com>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
"""
  Simple script wrapper

"""
import csv
import sys
from optparse import OptionParser
import itertools
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

try:
    from itertools import ifilter
except ImportError:
    ifilter = filter

# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)


__version__ = '0.5'

def conn_mysql(attrs):
    """ Connection for mysql database """

    try:
        import MySQLdb
        return MySQLdb.connect(host=attrs.get('host'), user=attrs.get('user'),
                               passwd=attrs.get('password'), port=int(attrs.get('port')),
                               db=attrs.get('dbname'))

    except ImportError:
        print(" MySQLdb driver not installed")
        sys.exit(2)


def conn_pgsql(attrs):
    """ Connection for postgresql database """
    dsn = " ".join(["%s=%s" % (k, v) for k, v in attrs.items()])
    try:
        import psycopg2
        return psycopg2.connect(dsn)
    except ImportError:
        try:  
            import psycopg
            return psycopg.connect(dsn)
        except ImportError:
            print("psycopg driver not installed")
            sys.exit(2)


def conn_sqlite(attrs):
    """ Connection for sqlite database """
    filename = attrs['dbname']
    try:
        import sqlite3
        return sqlite3.connect(filename)
    except ImportError:
        pass

    try:
        import sqlite
        return sqlite.connect(filename)
    except ImportError:
        pass

    try:
        from pysqlite2 import dbapi2 as sqlite
        return sqlite.connect(filename)
    except ImportError:
        pass

    print("sqlite driver not installed")
    sys.exit(2)

DRIVERS = {
    'mysql': conn_mysql,
    'pgsql': conn_pgsql,
    'sqlite': conn_sqlite,
}


def generate_sql_insert(tablename, columns, placeholder="'%s'"):
    """ Generate the sql statment template for all the inserts """
    fields = ",".join(columns)
    places = ",".join([placeholder] * len(columns))
    sql = "insert into %s (%s) values (%s);" % (tablename, fields, places)
    return sql


def prepare_values(values):
    """ Prepate the values for the SQL output """
    prepared = []
    for value in values:
        if value == None:
            prepared.append('')
        else:
            prepared.append(unicode(value).strip())
    return tuple(prepared)


def output_sql(outfile, table, columns, iter_input, placeholder="'%s'"):
    """ Output SQL to outfile reading input from iter_input """
    insert_sql = generate_sql_insert(table, columns, placeholder)
    for row in ifilter(None, iter_input):
        outfile.write(insert_sql % prepare_values(row) + "\n")


def output_sql_from_file(outfile, table, inputfile, placeholder="'%s'",
                          delimiter=','):
    """
        Output SQL to outfile reading input from inputfile
        Inputfile is a open file formated as CSV
    """
    input_ = csv.reader(inputfile, dialect='excel', delimiter=delimiter)
    columns = next(input_)
    return output_sql(outfile, table, columns, input_, placeholder)


def get_sql(tablename, inputfile, delimiter=';'):
    """
         Get all SQL as a single value
    """
    strout = StringIO()
    output_sql_from_file(strout, tablename, inputfile,
                                    delimiter=delimiter)
    return strout.getvalue()


def load(cursor, table, columns, input_, placeholder='?'):
    """
         Load a input iterator to a database cursor
    """
    insert_sql = generate_sql_insert(table, columns, placeholder)
    for row in ifilter(None, input_):
        cursor.execute(insert_sql, prepare_values(row))


def load_file(cursor, table, inputfile, placeholder='?', delimiter=','):
    """
         Load a CSV file to a database cursor
    """
    input_ = csv.reader(inputfile, dialect='excel', delimiter=delimiter)
    columns = next(input_)
    load(cursor, table, columns, input_, placeholder)


def load_file_copy(cursor, table, inputfile, delimiter=','):
    """
         Load a CSV file to dtabase using copy
    """
    columns = inputfile.readline().split(delimiter)
    cursor.copy_from(inputfile, table, sep=delimiter, columns=columns)


def parse_args(argv):
    """
       Parse and validate args
    """
    usage = "usage: %prog [options] filename.csv\n" + \
            "If no database provided, display SQL to stdout"

    parser = OptionParser(usage=usage, version=__version__)

    parser.add_option("-D", "--driver", dest="driver",
                      help="database driver [%s]" % ",".join(DRIVERS))
    parser.add_option("-H", "--hostname", dest="hostname",
                      help="database server hostname,defaults to localhost",
                      metavar="HOSTNAME")
    parser.add_option("-d", "--dbname", dest="dbname",
                      help="database name (filename on sqlite)")
    parser.add_option("-u", "--user", dest="user",
                      help="database username")
    parser.add_option("-p", "--password", dest="password",
                      help="database password")
    parser.add_option("-P", "--port", dest="port",
                      help="database port")
    parser.add_option("-t", "--table", dest="table",
                      help="database table to load")
    parser.add_option("", "--test", action="store_true", dest="test",
                      help="run text, do no commit to the database")
    parser.add_option("--delimiter", dest="delimiter", default=";",
                      help="CSV file field delimiter, by default semi-colon")
    parser.add_option("-o", "--output", dest="output", metavar="FILE",
                      help="Output file for SQL commands")
    parser.add_option("", "--copy", action="store_true", dest="copy",
                      help="Use copy to load to database")
    options, args = parser.parse_args(args=argv)

    if len(args) != 1:
        parser.error("Filename not provided")

    if not options.table:
        parser.error("table is required")

    if options.driver and options.driver not in DRIVERS:
        parser.error("database driver not suported")

    if options.driver and not options.dbname:
        parser.error("database is required")

    return options, args


def get_dsn_attrs(options):
    """ Get dsn_attrs from command line options """
    dsn_attrs = {}
    if options.hostname:
        dsn_attrs['host'] = options.hostname
    if options.dbname:
        dsn_attrs['dbname'] = options.dbname
    if options.password:
        dsn_attrs['password'] = options.password
    if options.user:
        dsn_attrs['user'] = options.user
    if options.port:
        dsn_attrs['port'] = options.port

    return dsn_attrs


def main(argv=None):
    """ Script entry point """
    if not argv:
        argv = sys.argv[1:]

    options, args = parse_args(argv)

    if options.driver:

        dsn_attrs = get_dsn_attrs(options)

        connection = DRIVERS[options.driver]
        conn = connection(dsn_attrs)
        cur = conn.cursor()

        with open(args[0]) as csvfile:
            if options.copy:
                load_file_copy(cur, options.table, csvfile,
                               delimiter=options.delimiter)
            else:
                if options.driver in ("mysql", "pgsql"):
                    load_file(cur, options.table, csvfile, placeholder="%s",
                              delimiter=options.delimiter)
                else:
                    load_file(cur, options.table, csvfile, placeholder="?",
                              delimiter=options.delimiter)

        if not options.test:
            conn.commit()
    else:

        with open(args[0]) as csvfile:
            if not options.output or options.output == "-":
                output_sql_from_file(sys.stdout, options.table, csvfile,
                                     delimiter=options.delimiter)
            else:
                out = open(options.output, "w")
                output_sql_from_file(out, options.table, csvfile,
                                     delimiter=options.delimiter)
                out.close()

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
