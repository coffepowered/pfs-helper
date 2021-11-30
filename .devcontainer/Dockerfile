FROM jupyter/datascience-notebook:ubuntu-20.04

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.

RUN pip install mdutils jupyter_contrib_nbextensions && jupyter contrib nbextension install --user
RUN pip install jupyter_nbextensions_configurator && jupyter nbextensions_configurator enable --user

