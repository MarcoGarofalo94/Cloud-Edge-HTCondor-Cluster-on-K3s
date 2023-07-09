FROM centos:8

RUN cd /etc/yum.repos.d/
RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
RUN sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
#Update
RUN yum update -y

# Dependencies
RUN yum -y install wget

#Install Python 3
RUN dnf install -y python3
RUN dnf install -y  python2 python2-chardet
RUN dnf install -y policycoreutils-python-utils


# Install yum repository
RUN cd /etc/yum.repos.d && wget https://research.cs.wisc.edu/htcondor/yum/repo.d/htcondor-stable-rhel8.repo

# Import signing key
RUN wget http://research.cs.wisc.edu/htcondor/yum/RPM-GPG-KEY-HTCondor && rpm --import RPM-GPG-KEY-HTCondor

# Install HTCondor
RUN yum -y install condor

# Install Nano
RUN yum -y install nano

# Install Java 1.8.0
RUN yum install -y java-1.8.0-openjdk-devel



#Install pegasus
RUN curl --output /etc/yum.repos.d/pegasus.repo https://download.pegasus.isi.edu/pegasus/rhel/8/pegasus.repo
RUN dnf install -y epel-release
RUN dnf install -y --enablerepo powertools pegasus

COPY condor_config.local /etc/condor/
COPY central-manager /etc/condor/config.d
COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
