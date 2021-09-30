#!venv/bin/python
import os
from flask import Flask, url_for, redirect, render_template, request, abort
from flask_sqlalchemy import SQLAlchemy
from flask_security import Security, SQLAlchemyUserDatastore, \
    UserMixin, RoleMixin, login_required, current_user
from flask_security.utils import encrypt_password
import flask_admin
from flask_admin.contrib import sqla
from flask_admin import helpers as admin_helpers
from flask_admin import BaseView, expose
from wtforms import PasswordField
import pandas as pd

# Create Flask application
app = Flask(__name__)
app.config.from_pyfile('config.py')
db = SQLAlchemy(app)


# Define models
roles_users = db.Table(
    'roles_users',
    db.Column('user_id', db.Integer(), db.ForeignKey('user.id')),
    db.Column('role_id', db.Integer(), db.ForeignKey('role.id'))
)


class Role(db.Model, RoleMixin):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(80), unique=True)
    description = db.Column(db.String(255))

    def __str__(self):
        return self.name


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    last_name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    active = db.Column(db.Boolean())
    confirmed_at = db.Column(db.DateTime())
    roles = db.relationship('Role', secondary=roles_users,
                            backref=db.backref('users', lazy='dynamic'))

    def __str__(self):
        return self.email


# Setup Flask-Security
user_datastore = SQLAlchemyUserDatastore(db, User, Role)
security = Security(app, user_datastore)


# Create customized model view class
class MyModelView(sqla.ModelView):

    def is_accessible(self):
        if not current_user.is_active or not current_user.is_authenticated:
            return False

        if current_user.has_role('superuser'):
            return True

        return False

    def _handle_view(self, name, **kwargs):
        """
        Override builtin _handle_view in order to redirect users when a view is not accessible.
        """
        if not self.is_accessible():
            if current_user.is_authenticated:
                # permission denied
                abort(403)
            else:
                # login
                return redirect(url_for('security.login', next=request.url))


    # can_edit = True
    edit_modal = True
    create_modal = True
    can_export = True
    can_view_details = True
    details_modal = True

class UserView(MyModelView):
    column_editable_list = ['email', 'first_name', 'last_name']
    column_searchable_list = column_editable_list
    column_exclude_list = ['password']
    #form_excluded_columns = column_exclude_list
    column_details_exclude_list = column_exclude_list
    column_filters = column_editable_list
    form_overrides = {
        'password': PasswordField
    }



# Flask views
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/details')
@login_required
def details():
    return render_template('pages/details.html')


@app.context_processor
def context_processor():
    df = pd.read_csv('output/user_pc_ct.csv')
    df2=df.drop_duplicates()
    legend1 = 'USER vs Async Score(PC Count)'
    labels1 = list(df2.user)
    values1 = list(df2.ascore)
    anamoly_pc_user = list(df2['user'].loc[df2['ascore']<0])
    pc_count= len(df2.loc[df2['ascore'] < 0 ])
    # count1 =len((df2.threat)== "[-1]")
    # values2 = list(df.threat)

    df = pd.read_csv('output/device_file_full_result.csv')
    df2 = df.drop_duplicates()
    legend2 = 'USER vs Async Score(Device)'
    labels2 = list(df2.user)
    values2 = list(df2.ascore)
    anamoly_device = list(df2['user'].loc[df2['ascore']<0])
    device_count= len(df2.loc[df2['ascore'] < 0 ])
    # values4 = list(df.threat)

    df = pd.read_csv('output/psychometric_result.csv')
    df2 = df.drop_duplicates()
    legend3 = 'USER vs Async Score(psychometric)'
    labels3 = list(df2.user)
    values3 = list(df2.ascore)
    anamoly_psychometric = list(df2['user'].loc[df2['ascore']<0])
    psychometric_count= len(df2.loc[df2['ascore'] < 0 ])

    df = pd.read_csv('output/user_log_result.csv')
    df2 = df.drop_duplicates()
    legend4 = 'USER vs Async Score(User logon)'
    labels4 = list(df2.user)
    values4 = list(df2.ascore)
    anamoly_user_logon = list(df2['user'].loc[df2['ascore']<0])
    User_logon_count= len(df2.loc[df2['ascore'] < 0 ])
    

    df = pd.read_csv('output/all_parameters_result.csv')
    legend5 = 'USER vs Async Score(all parameter))'
    labels5 = list(df2.user)
    values5 = list(df2.ascore)
    anamoly_email = list(df2['user'].loc[df2['ascore']<0])
    email_count= len(df2.loc[df2['ascore'] < 0 ])
    anamoly_All = list(df2['user'].loc[df2['ascore']<0])
    All_count= len(df2.loc[df2['ascore'] < 0 ])
    

    df = pd.read_csv('output/email_result.csv')
    df2 = df.drop_duplicates()
    legend6 = 'USER vs Async Score(email)'
    labels6 = list(df2.user)
    values6 = list(df2.ascore)
    anamoly_email = list(df2['user'].loc[df2['ascore']<0])
    email_count= len(df2.loc[df2['ascore'] < 0 ])


    return dict(
        values1=values1, labels1=labels1,legend1=legend1,pc_count=pc_count,anamoly_pc_user=anamoly_pc_user,
        values2=values2,labels2=labels2,legend2=legend2,device_count= device_count,anamoly_device=anamoly_device,
        values3=values3,labels3=labels3,legend3=legend3,psychometric_count=psychometric_count,anamoly_psychometric=anamoly_psychometric,
        values4=values4,labels4=labels4,legend4=legend4,User_logon_count=User_logon_count,anamoly_user_logon=anamoly_user_logon,
        values5=values5, labels5=labels5,legend5=legend5,All_count=All_count,anamoly_All=anamoly_All,
        values6=values6, labels6=labels6,legend6=legend6,email_count=email_count,anamoly_email=anamoly_email
    )



# Create admin
admin = flask_admin.Admin(
    app,
    'DARKS',
    base_template='my_master.html',
    template_mode='bootstrap4',
)

# Add model views
admin.add_view(MyModelView(Role, db.session, menu_icon_type='fa', menu_icon_value='fa-server', name="System Roles"))
admin.add_view(UserView(User, db.session, menu_icon_type='fa', menu_icon_value='fa-users', name="System Users"))
# admin.add_view(CustomView(name="Custom view", endpoint='custom', menu_icon_type='fa', menu_icon_value='fa-connectdevelop',))

# define a context processor for merging flask-admin's template context into the
# flask-security views.
@security.context_processor
def security_context_processor():
    return dict(
        admin_base_template=admin.base_template,
        admin_view=admin.index_view,
        h=admin_helpers,
        get_url=url_for
    )

def build_sample_db():
    """
    Populate a small db with some example entries.
    """

    import string
    import random

    db.drop_all()
    db.create_all()

    with app.app_context():
        user_role = Role(name='user')
        super_user_role = Role(name='superuser')
        db.session.add(user_role)
        db.session.add(super_user_role)
        db.session.commit()

        test_user = user_datastore.create_user(
            first_name='Admin',
            email='admin',
            password=encrypt_password('admin'),
            roles=[user_role, super_user_role]
        )

        first_names = []

        last_names = []

        for i in range(len(first_names)):
            tmp_email = first_names[i].lower() + "." + last_names[i].lower() + "@example.com"
            tmp_pass = ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(10))
            user_datastore.create_user(
                first_name=first_names[i],
                last_name=last_names[i],
                email=tmp_email,
                password=encrypt_password(tmp_pass),
                roles=[user_role, ]
            )
        db.session.commit()
    return

if __name__ == '__main__':

    # Build a sample db on the fly, if one does not exist yet.
    app_dir = os.path.realpath(os.path.dirname(__file__))
    database_path = os.path.join(app_dir, app.config['DATABASE_FILE'])
    if not os.path.exists(database_path):
        build_sample_db()

    # Start app
    app.run(debug=True)
