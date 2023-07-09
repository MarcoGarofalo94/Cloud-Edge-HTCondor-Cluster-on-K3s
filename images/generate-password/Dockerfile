FROM centos:8

RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
#Update
RUN yum update -y

# Dependencies
RUN yum -y install wget

# Install yum repository
RUN cd /etc/yum.repos.d && wget http://research.cs.wisc.edu/htcondor/yum/repo.d/htcondor-stable-rhel8.repo

# Import signing key
RUN wget http://research.cs.wisc.edu/htcondor/yum/RPM-GPG-KEY-HTCondor && rpm --import RPM-GPG-KEY-HTCondor

# Install HTCondor
RUN yum -y install condor

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
